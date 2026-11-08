import csv
import json

import pytest

from small_object_lab.research.selector import (
    FEATURE_NAMES,
    LearnedSelector,
    train_selector,
)


SR = "sr_p020_x4"
GT_FIELDS = (
    "session_id", "frame_id", "object_id", "label", "visible",
    "x", "y", "width", "height", "reviewed",
)


def features(width=0.05):
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values.update(width_frac=width, height_frac=0.1, confidence=0.8)
    return values


def detection(*, x=10, label="cup", confidence=0.4):
    return dict(x=x, y=20, width=30, height=40, label=label, confidence=confidence)


def candidate(action="none", *, correct=True, confidence=0.4, elapsed=0):
    return {
        "action_id": action, "kind": "none" if action == "none" else "sr",
        "padding": 0 if action == "none" else 0.2,
        "scale": 1 if action == "none" else 4, "elapsed_ms": elapsed,
        "detection": detection(x=10 if correct else 100, confidence=confidence),
    }


def row(session="train-a", frame=1, width=0.05, sr_correct=True):
    return {
        "session_id": session, "frame_id": frame, "timestamp_s": frame / 10,
        "features": features(width),
        "candidates": [
            candidate(correct=not sr_correct, confidence=0.99),
            candidate(SR, correct=sr_correct, confidence=0.3, elapsed=20),
        ],
    }


def annotation(sample, *, reviewed=True, visible=True):
    return dict(
        session_id=sample["session_id"], frame_id=sample["frame_id"], object_id="cup-1",
        label="cup", visible=int(visible), x=10, y=20, width=30, height=40,
        reviewed=int(reviewed),
    )


def inputs(tmp_path, samples, *, split=None, annotations=None):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(json.dumps(item) for item in samples), encoding="utf-8")
    annotations_path = tmp_path / "annotations.csv"
    with annotations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GT_FIELDS)
        writer.writeheader()
        writer.writerows(annotations if annotations is not None else [annotation(s) for s in samples])
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split or {
        "train": ["train-a"], "validation": [], "test": [],
    }), encoding="utf-8")
    return samples_path, annotations_path, split_path, tmp_path / "selector.json"


def test_train_load_roundtrip_learns_size_dependent_action_from_ground_truth(tmp_path):
    samples = [row(frame=i, width=0.05 if i % 2 else 0.2, sr_correct=bool(i % 2))
               for i in range(1, 41)]
    samples += [row("validation-a", 1, 0.05), row("validation-a", 2, 0.2, False)]
    paths = inputs(tmp_path, samples, split={
        "train": ["train-a"], "validation": ["validation-a"], "test": [],
    })
    report = train_selector(*paths)
    model = LearnedSelector.load(paths[-1])
    assert model.choose(features(0.05), ["none", SR]) == SR
    assert model.choose(features(0.2), [SR, "none"]) == "none"
    assert model.training_session_ids == {"train-a"}
    assert report["validation"]["visible_match_rate"] == 1
    assert report["train_sample_count"] == 40
    assert report["action_sample_counts"] == {"none": 40, SR: 40}
    assert not list(tmp_path.glob("*.tmp"))


def test_high_confidence_wrong_detection_is_not_a_positive_label(tmp_path):
    samples = [row(frame=i) for i in range(1, 6)]
    paths = inputs(tmp_path, samples)
    train_selector(*paths)
    model = LearnedSelector.load(paths[-1])
    # Full-frame confidence is .99, SR confidence .3; only SR overlaps GT.
    assert model.choose(features(), ["none", SR]) == SR


def test_matching_box_with_wrong_class_is_not_correct(tmp_path):
    sample = row(sr_correct=False)
    sample["candidates"][1]["detection"] = detection(label="bottle", confidence=1.0)
    paths = inputs(tmp_path, [sample])
    train_selector(*paths)
    assert LearnedSelector.load(paths[-1]).choose(features(), ["none", SR]) == "none"


def test_missing_candidates_are_not_scored_as_failures_and_unknown_actions_are_ignored(tmp_path):
    samples = [row(frame=i) for i in range(1, 11)]
    for sample in samples[2:]:
        sample["candidates"] = sample["candidates"][:1]
    paths = inputs(tmp_path, samples)
    report = train_selector(*paths)
    model = LearnedSelector.load(paths[-1])
    assert report["action_sample_counts"][SR] == 2
    assert model.choose(features(), ["none", SR]) == SR
    assert model.choose(features(), ["none", "untrained-scale"]) == "none"
    with pytest.raises(ValueError, match="No available action"):
        model.choose(features(), ["untrained-scale"])


def test_latency_cost_can_outweigh_detection_improvement(tmp_path):
    sample = row()
    sample["candidates"][1]["elapsed_ms"] = 1000
    paths = inputs(tmp_path, [sample])
    train_selector(*paths)
    assert LearnedSelector.load(paths[-1]).choose(features(), ["none", SR]) == "none"


def test_validation_does_not_change_scaler_and_test_outcomes_are_never_used(tmp_path):
    samples = [row(), row("validation-a", 1, width=1000, sr_correct=False)]
    test_sample = row("test-a", 1)
    test_sample["features"] = "must not be transformed"
    test_sample["candidates"] = None
    samples.append(test_sample)
    paths = inputs(tmp_path, samples, annotations=[annotation(s) for s in samples[:2]], split={
        "train": ["train-a"], "validation": ["validation-a"], "test": ["test-a"],
    })
    report = train_selector(*paths)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert payload["feature_mean"][0] == pytest.approx(0.05)
    assert payload["training_session_ids"] == ["train-a"]
    assert payload["actions"][SR]["sample_count"] == 1
    assert report["test_samples_not_used"] == 1
    assert report["validation"]["visible_match_rate"] == 0


@pytest.mark.parametrize("split", [
    {"train": ["train-a"], "validation": ["train-a"], "test": []},
    {"train": ["train-a", "train-a"], "validation": [], "test": []},
    {"train": ["other"], "validation": [], "test": []},
])
def test_rejects_overlapping_duplicate_and_unassigned_session_splits(tmp_path, split):
    paths = inputs(tmp_path, [row()], split=split)
    with pytest.raises(ValueError, match="Duplicate|Unassigned"):
        train_selector(*paths)
    assert not paths[-1].exists()


def test_unreviewed_annotations_are_not_training_labels(tmp_path):
    samples = [row(), row(frame=2, sr_correct=False)]
    paths = inputs(tmp_path, samples, annotations=[
        annotation(samples[0]), annotation(samples[1], reviewed=False),
    ])
    report = train_selector(*paths)
    assert report["train_sample_count"] == 1
    assert report["train_skipped_unreviewed_or_missing"] == 1
    assert LearnedSelector.load(paths[-1]).choose(features(), ["none", SR]) == SR


def test_training_requires_real_reviewed_annotations(tmp_path):
    sample = row()
    paths = inputs(tmp_path, [sample], annotations=[annotation(sample, reviewed=False)])
    with pytest.raises(ValueError, match="No reviewed train"):
        train_selector(*paths)
    assert not paths[-1].exists()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None, "bad"])
def test_rejects_nonfinite_or_invalid_features_at_train_and_inference(tmp_path, invalid):
    paths = inputs(tmp_path, [row()])
    train_selector(*paths)
    model = LearnedSelector.load(paths[-1])
    bad = features()
    bad["width_frac"] = invalid
    with pytest.raises(ValueError, match="finite"):
        model.choose(bad, ["none", SR])
    sample = row()
    sample["features"] = bad
    paths = inputs(tmp_path, [sample])
    with pytest.raises(ValueError, match="finite"):
        train_selector(*paths)


def test_rejects_duplicate_frames_and_duplicate_candidates(tmp_path):
    paths = inputs(tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="Duplicate sample"):
        train_selector(*paths)
    sample = row()
    sample["candidates"].append(sample["candidates"][0])
    paths = inputs(tmp_path, [sample])
    with pytest.raises(ValueError, match="Duplicate candidate"):
        train_selector(*paths)


def test_model_json_is_validated_before_inference(tmp_path):
    paths = inputs(tmp_path, [row()])
    train_selector(*paths)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["feature_scale"][0] = 0
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        LearnedSelector.load(paths[-1])
