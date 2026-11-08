import csv

import pytest

from small_object_lab.research.annotation import AnnotationReview
from small_object_lab.research.evaluation import evaluate_predictions, load_annotations
from small_object_lab.research.runner import ExperimentConfig, run_experiment
from small_object_lab.research.sources import create_annotation_template
from test_research_runner import FakeBackend, recording


def test_browsing_does_not_create_ground_truth_and_explicit_edits_resume(tmp_path):
    source = recording(tmp_path)
    output = tmp_path / "annotations.csv"
    review = AnnotationReview(source, output)
    review.save()
    assert not any(row["reviewed"] for row in load_annotations(output).values())
    review.commit(1, box=(20, 20, 12, 20))
    review.commit(2, visible=False)
    resumed = AnnotationReview(source, output)
    assert resumed.rows[1]["reviewed"] == 1
    assert resumed.rows[2]["visible"] == 0
    assert resumed.rows[3]["reviewed"] == 0
    with pytest.raises(ValueError, match="inside"):
        resumed.commit(3, box=(90, 20, 20, 20))


def test_generated_template_can_be_evaluated_without_fabricated_accuracy(tmp_path):
    source = recording(tmp_path)
    output = tmp_path / "annotations.csv"
    create_annotation_template(source, output)
    run = tmp_path / "run"
    run_experiment(source, run, FakeBackend(), ExperimentConfig(), methods=("baseline",))
    result = evaluate_predictions(run / "predictions.csv", output)
    assert result["groups"][0]["evaluation_status"] == "no_reviewed_annotations"
    assert result["groups"][0]["detection"]["recall"] is None
    review = AnnotationReview(source, output)
    for frame_id in review.rows:
        review.commit(frame_id, box=(20, 20, 12, 20))
    result = evaluate_predictions(run / "predictions.csv", output)
    assert result["groups"][0]["detection"]["recall"] == 1.0
    with output.open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 4
