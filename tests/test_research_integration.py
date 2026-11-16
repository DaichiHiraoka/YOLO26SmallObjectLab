import csv
import json
import time

import pytest

from small_object_lab.detector import LabSettings, SmallObjectLab
from small_object_lab.metrics import Detection
from small_object_lab.research.annotation import AnnotationReview
from small_object_lab.research.dataset import assemble_dataset
from small_object_lab.research.runner import ExperimentConfig, run_experiment
from small_object_lab.research.selector import LearnedSelector, train_selector
from test_research_runner import FakeBackend, recording


def test_candidate_collection_annotation_assembly_train_and_held_out_replay(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    source = recording(root)
    run = root / "run"
    run_experiment(source, run, FakeBackend(), ExperimentConfig(), methods=("baseline",), collect_candidates=True)
    gt = root / "annotations.csv"
    review = AnnotationReview(source, gt)
    for frame_id in review.rows:
        review.commit(frame_id, box=(20, 20, 12, 20))
    dataset = tmp_path / "dataset"
    report = assemble_dataset([run], [gt], dataset)
    assert report["reviewed_frames"] == 4
    split = dataset / "split.json"
    split.write_text(json.dumps({"train": [source.session_id], "validation": [], "test": ["held-out"]}))
    model_path = dataset / "selector.json"
    train_selector(dataset / "samples.jsonl", dataset / "annotations.csv", split, model_path, frames="all")
    model = LearnedSelector.load(model_path)
    assert model.payload["experiment_config"]["confidence"] == 0.2
    other = tmp_path / "test"
    other.mkdir()
    test_source = recording(other)
    # Different synthetic session identity; for software validation only.
    test_source.session_id = "held-out"
    final = run_experiment(test_source, other / "run", FakeBackend(), ExperimentConfig(), methods=("learned",), selector=model)
    assert final["status"] == "complete"
    with pytest.raises(ValueError, match="configuration"):
        run_experiment(test_source, other / "bad", FakeBackend(), ExperimentConfig(confidence=0.1), methods=("learned",), selector=model)
    with pytest.raises(FileExistsError):
        train_selector(dataset / "samples.jsonl", dataset / "annotations.csv", split, model_path, frames="all")
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_dataset([run, run], [gt], tmp_path / "duplicate")


def test_measurements_use_displayed_largest_not_smallest_or_stale_target(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path)
    small = Detection("cup", 0.9, 10, 10, 5, 5, streak=9)
    large = Detection("cup", 0.6, 20, 20, 30, 40, streak=1)
    lab.latest_detections = [small, large]
    lab.smallest_current = small
    lab.smallest_confirmed = small
    lab.frame_width = lab.frame_height = 100
    lab.record_measurement({})
    lab.next_auto_save_at = time.monotonic() - 1
    lab._maybe_auto_save(frame_number=1, detections=[small, large], confirmed=[small],
                         settings=LabSettings(), jpeg_payload=b"test")
    with lab.measurements_path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["width_px"] for r in rows] == ["30", "30"]
    assert all(r["detection_status"] == "unstable" for r in rows)
    lab.latest_detections = []
    with pytest.raises(ValueError):
        lab.record_measurement({})
    lab.stop()
