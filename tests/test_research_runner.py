import csv
import json

import cv2
import numpy as np
import pytest

from small_object_lab.metrics import Detection
from small_object_lab.research.runner import (Action, ExperimentConfig, MethodEngine,
                                            available_actions, execute_action, run_experiment)
from small_object_lab.research.sources import FrameSource, create_annotation_template
from small_object_lab.research.tracking import TargetTracker


def box(x=20, width=12, confidence=0.8):
    return Detection("cup", confidence, x, 20, width, 20)


def recording(tmp_path, n=4):
    folder = tmp_path / "recording"
    folder.mkdir()
    (folder / "frames").mkdir()
    with (folder / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_id", "timestamp_s", "path", "width", "height"])
        writer.writeheader()
        for index in range(1, n + 1):
            path = f"frames/{index:08d}.png"
            cv2.imencode(".png", np.full((100, 100, 3), index, dtype=np.uint8))[1].tofile(folder / path)
            writer.writerow(dict(frame_id=index, timestamp_s=(index-1)*0.1, path=path, width=100, height=100))
    (folder / "manifest.json").write_text(json.dumps(dict(schema_version=1, session_id="synthetic-test",
          format="png_sequence", complete=True, dropped_frames=0, errors=[])))
    return FrameSource(folder)


class FakeBackend:
    def __init__(self):
        self.full_calls = 0
        self.local_thresholds = []

    def synchronize(self):
        pass

    def warmup(self, *_args, **_kwargs):
        pass

    def detect(self, frame, config, *, local=False):
        if not local:
            self.full_calls += 1
            return [box()]
        self.local_thresholds.append(config.confidence)
        return []

    def enhance(self, crop, action):
        return cv2.resize(crop, None, fx=action.scale, fy=action.scale)


def test_identity_does_not_jump_to_larger_object_and_expires_by_time():
    tracker = TargetTracker(0.75)
    tracker.update(box(), 0.0)
    tracker.update(box(22), 0.1)
    assert tracker.match([box(25), box(300, width=200)], 0.2).x == 25
    assert tracker.track_id == 1
    assert tracker.predict(0.5).x > 22
    assert tracker.predict(0.9) is None
    tracker.update(box(300), 0.9)
    assert tracker.track_id == 2


def test_sr_recovery_refreshes_target_and_is_observed():
    class Recover(FakeBackend):
        def detect(self, frame, config, *, local=False):
            return [Detection("cup", 0.9, 32, 32, 48, 80)]
    engine = MethodEngine("fixed", ExperimentConfig())
    engine.tracker.update(box(), 0.0)
    detection, _, attempts, _ = engine.process(Recover(), np.zeros((100, 100, 3), np.uint8), [], 0.6)
    assert detection is not None and attempts[0]["adopted"]
    assert engine.tracker.predict(1.0) is not None


def test_comparison_uses_same_frame_indices_and_thresholds(tmp_path):
    source, backend = recording(tmp_path), FakeBackend()
    output = tmp_path / "result"
    result = run_experiment(source, output, backend, ExperimentConfig(), collect_candidates=True)
    assert backend.full_calls == 4
    assert result["status"] == "complete"
    with (output / "predictions.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    for method in ("baseline", "fixed", "bicubic", "rule"):
        assert [r["frame_id"] for r in rows if r["method"] == method] == ["1", "2", "3", "4"]
    assert set(backend.local_thresholds) == {0.2}
    samples = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(samples) == 4
    # none + 17 feasible SR (x8 with 300% padding exceeds the 640 px budget) + 6 crop-only
    assert len(samples[0]["candidates"]) == 24
    assert samples[0]["candidates"][0]["action_id"] == "none"
    with pytest.raises(FileExistsError):
        run_experiment(source, output, backend, ExperimentConfig())


def test_missing_detection_is_recorded_not_hallucinated_from_prediction(tmp_path):
    class Missing(FakeBackend):
        def detect(self, frame, config, *, local=False):
            if local:
                return []
            self.full_calls += 1
            return [box()] if self.full_calls == 1 else []
    output = tmp_path / "out"
    run_experiment(recording(tmp_path), output, Missing(), ExperimentConfig(), methods=("baseline", "fixed"))
    with (output / "predictions.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    missing = [r for r in rows if r["frame_id"] == "2"]
    assert len(missing) == 2
    assert all(r["observed"] == "0" and r["x"] == "" for r in missing)


def test_action_budget_and_input_size_do_not_claim_sr_scale_is_yolo_magnification():
    backend, image, config = FakeBackend(), np.zeros((100, 100, 3), np.uint8), ExperimentConfig()
    x2 = execute_action(backend, image, box(), box(), Action("sr", 0.2, 2), config)
    x8 = execute_action(backend, image, box(), box(), Action("sr", 0.2, 8), config)
    assert x2["yolo_target_width_px"] == x8["yolo_target_width_px"]
    large = available_actions(box(width=600), (720, 720, 3), config)
    assert large[0].action_id == "none"
    # No enlargement fits the 640 px budget; only the six crop-only (x1) actions remain.
    assert [a.kind for a in large[1:]] == ["crop"] * 6


def test_template_is_unreviewed_and_incomplete_recording_rejected(tmp_path):
    source = recording(tmp_path)
    path = tmp_path / "annotations.csv"
    assert create_annotation_template(source, path) == 4
    with path.open(encoding="utf-8-sig") as stream:
        assert all(r["reviewed"] == "0" and r["visible"] == "" for r in csv.DictReader(stream))
    manifest = source.path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["dropped_frames"] = 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        FrameSource(source.path)


def test_training_session_replay_requires_explicit_diagnostic_override(tmp_path):
    class Selector:
        training_session_ids = {"synthetic-test"}
        def choose(self, _features, _available):
            return "none"
    with pytest.raises(ValueError, match="used to train"):
        run_experiment(recording(tmp_path), tmp_path / "out", FakeBackend(), ExperimentConfig(),
                       methods=("learned",), selector=Selector())


def test_part_of_the_object_is_not_adopted_and_tracker_accepts_enclosing_box():
    # Observed key bunch at (100, 100, 70, 54). A zoomed crop finds only the fob.
    bunch = Detection("key", 0.80, 100, 100, 70, 54)

    class PartOnly(FakeBackend):
        def __init__(self, local):
            super().__init__()
            self.local = local

        def detect(self, frame, config, *, local=False):
            return [self.local] if local else []

    frame = np.zeros((360, 360, 3), np.uint8)
    action = Action("crop", 1.0, 1)  # region starts at (30, 46) for this bunch
    fob = Detection("key", 0.95, 100 - 30 + 50, 100 - 46 + 30, 16, 12)
    result = execute_action(PartOnly(fob), frame, bunch, bunch, action, ExperimentConfig(label="key"))
    assert result["local_detection"] is not None and not result["adopted"]
    whole = Detection("key", 0.90, 100 - 30 + 2, 100 - 46 + 1, 68, 54)
    result = execute_action(PartOnly(whole), frame, bunch, bunch, action, ExperimentConfig(label="key"))
    assert result["adopted"]

    tracker = TargetTracker(0.75)
    tracker.update(Detection("key", 0.9, 150, 130, 16, 12), 0.0)  # box had shrunk to the fob
    assert tracker.match([bunch], 0.05) == bunch


def test_live_merge_rejects_a_part_of_the_target():
    from small_object_lab.detector import SmallObjectLab

    bunch = Detection("key", 0.80, 100, 100, 70, 54)
    fob = Detection("key", 0.95, 150, 130, 16, 12, enhanced=True, enhancement_scale=4)
    merged, used = SmallObjectLab._merge_super_resolution_detection([bunch], bunch, fob)
    assert not used and merged == [bunch]
    whole = Detection("key", 0.90, 101, 100, 69, 54, enhanced=True, enhancement_scale=4)
    merged, used = SmallObjectLab._merge_super_resolution_detection([bunch], bunch, whole)
    assert used and merged == [whole]
