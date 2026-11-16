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


def test_drafts_track_one_object_fall_back_to_crop_and_interpolate_short_gaps():
    import numpy as np
    from small_object_lab.research.proposals import propose_frames

    calls = []

    def predict(image, imgsz):
        calls.append(imgsz)
        frame = int(image[0, 0, 0])
        if imgsz == 640:  # crop around the last draft finds the small object (crop coordinates)
            return [(75.0, 75.0, 10.0, 10.0, 0.4)] if frame == 3 else []
        if frame in (1, 2):
            # a more confident decoy far away must not replace the tracked object
            return [(200.0, 200.0, 10.0, 10.0, 0.6)] + ([(20.0, 20.0, 40.0, 40.0, 0.9)] if frame == 2 else [])
        return [(202.0, 200.0, 10.0, 10.0, 0.7)] if frame == 6 else []

    frames = [(i, np.full((300, 300, 3), i, np.uint8)) for i in range(1, 8)]
    drafts = propose_frames(frames, predict, max_gap=2)
    assert drafts[2].box == (200, 200, 10, 10)
    # 160 px crop centred on the last draft starts at (125, 125)
    assert drafts[3].source == "crop" and drafts[3].box == (200, 200, 10, 10)
    assert drafts[4].source == drafts[5].source == "interpolated"
    assert drafts[5].box == (201, 200, 10, 10)
    assert 7 not in drafts and 640 in calls


def test_grid_accepts_and_rejects_drafts_with_provenance(tmp_path):
    from small_object_lab.research.annotation import apply_grid_page, grid_queue
    from small_object_lab.research.proposals import Proposal

    source = recording(tmp_path)
    output = tmp_path / "annotations.csv"
    review = AnnotationReview(source, output)
    drafts = {1: Proposal(1, (20, 20, 12, 20), 0.9, "full"), 2: Proposal(2, (20, 20, 12, 20), 0.5, "interpolated")}
    frames = [{"frame_id": i} for i in (1, 2, 3, 4)]
    assert grid_queue(review, frames, drafts) == [1, 2]
    apply_grid_page(review, [1, 2], drafts, rejected={2})
    resumed = AnnotationReview(source, output)
    assert resumed.rows[1]["reviewed"] == 1 and resumed.rows[2]["reviewed"] == 0
    assert resumed.provenance == {1: "accepted_draft_full"}
    assert resumed.rejected == {2}
    # rejected drafts and undrafted frames are left for one-by-one review, not the grid
    assert grid_queue(resumed, frames, drafts) == []
    resumed.commit(2, visible=False)
    assert AnnotationReview(source, output).provenance[2] == "invisible"
    assert load_annotations(output)[("synthetic-test", 1)]["reviewed"]


def test_absence_grid_confirms_empty_frames_and_flags_visible_ones(tmp_path):
    from small_object_lab.research.annotation import absence_queue, apply_absence_page, apply_grid_page
    from small_object_lab.research.proposals import Proposal

    source = recording(tmp_path)
    output = tmp_path / "annotations.csv"
    review = AnnotationReview(source, output)
    drafts = {1: Proposal(1, (20, 20, 12, 20), 0.9, "full"), 2: Proposal(2, (20, 20, 12, 20), 0.5, "full")}
    frames = [{"frame_id": i} for i in (1, 2, 3, 4)]
    apply_grid_page(review, [1, 2], drafts, rejected={2})
    # accepted drafts are done; a rejected draft and undrafted frames get the whole-frame check
    assert absence_queue(review, frames, drafts) == [2, 3, 4]
    apply_absence_page(review, [2, 3, 4], has_object={4})
    resumed = AnnotationReview(source, output)
    assert resumed.rows[2]["reviewed"] == 1 and resumed.rows[2]["visible"] == 0
    assert resumed.provenance[3] == "grid_not_visible" and resumed.rows[4]["reviewed"] == 0
    assert resumed.flagged == {4} and absence_queue(resumed, frames, drafts) == []
    # going back to a confirmed page and flagging a frame reopens it for drawing
    apply_absence_page(resumed, [3], has_object={3})
    assert resumed.rows[3]["reviewed"] == 0 and 3 not in resumed.provenance and resumed.flagged == {3, 4}
    apply_absence_page(resumed, [3], has_object=set())
    resumed.commit(4, box=(20, 20, 12, 20))
    final = AnnotationReview(source, output)
    assert final.flagged == set() and final.provenance[4] == "manual_box"
    final.commit(4, visible=False)
    assert AnnotationReview(source, output).provenance[4] == "invisible"
