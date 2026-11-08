import csv
import json

import pytest

from small_object_lab.research.evaluation import (
    ANNOTATION_FIELDS,
    PREDICTION_FIELDS,
    box_iou,
    evaluate_predictions,
    load_annotations,
)


def annotation(frame, **changes):
    row = dict(session_id="s1", frame_id=frame, object_id="cup1", label="cup",
               visible=1, x=10, y=20, width=20, height=30, reviewed=1)
    row.update(changes)
    return row


def prediction(frame, **changes):
    row = dict(session_id="s1", method="baseline", frame_id=frame, timestamp_s=frame * 0.1,
               track_id="1", label="cup", confidence=0.8, x=10, y=20, width=20, height=30,
               observed=1, inference_ms=10, action_id="none")
    row.update(changes)
    return row


def miss(frame, **changes):
    return prediction(frame, observed=0, track_id="", label="", confidence="",
                      x="", y="", width="", height="", **changes)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def evaluate(tmp_path, predictions, annotations):
    pred_path = write_csv(tmp_path / "predictions.csv", PREDICTION_FIELDS, predictions)
    gt_path = write_csv(tmp_path / "annotations.csv", ANNOTATION_FIELDS, annotations)
    result = evaluate_predictions(pred_path, gt_path)
    json.dumps(result, allow_nan=False)
    return result


def test_evaluation_scores_wrong_box_label_and_invisible_false_positives(tmp_path):
    result = evaluate(tmp_path, [prediction(0), prediction(1, x=100),
                                prediction(2, label="bottle"), prediction(3), miss(4)],
                      [annotation(0), annotation(1), annotation(2),
                       annotation(3, visible=0, x="", y="", width="", height=""),
                       annotation(4, visible=0)])
    metrics = result["groups"][0]["detection"]
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 3
    assert metrics["false_negatives"] == 2
    assert metrics["true_negative_frames"] == 1
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["precision"] == 0.25
    assert metrics["mean_matched_iou"] == 1


def test_unknown_and_unreviewed_frames_are_excluded_and_break_loss_runs(tmp_path):
    result = evaluate(tmp_path, [miss(0), prediction(1, x=200), prediction(2, x=200), miss(3), prediction(4)],
                      [annotation(0), annotation(2, reviewed=0, x="", y="", width="", height=""),
                       annotation(3), annotation(4)])
    group = result["groups"][0]
    assert group["coverage"]["prediction_frames_without_annotation"] == 1
    assert group["coverage"]["prediction_frames_unreviewed"] == 1
    assert group["coverage"]["reviewed_fraction_of_predictions"] == 0.6
    assert group["detection"]["false_positives"] == 0
    assert group["detection"]["recall"] == pytest.approx(1 / 3)
    assert group["tracking"]["lost_runs"] == 2
    assert group["tracking"]["censored_runs"] == 1
    assert group["tracking"]["reacquisition_time_s"]["p50"] == pytest.approx(0.1)


def test_occlusion_is_not_detection_loss_and_id_switch_remains_visible(tmp_path):
    result = evaluate(tmp_path, [prediction(0), miss(1), miss(2), prediction(3, track_id="2")],
                      [annotation(0), annotation(1, visible=0), annotation(2, visible=0), annotation(3)])
    group = result["groups"][0]
    assert group["detection"]["recall"] == 1
    assert group["tracking"]["lost_runs"] == 0
    assert group["tracking"]["id_switches"] == 1
    assert group["tracking"]["dominant_id_fraction_of_matches"] == 0.5


def test_motion_only_box_is_not_a_detection(tmp_path):
    result = evaluate(tmp_path, [prediction(0, observed=0), prediction(1, observed=0)],
                      [annotation(0), annotation(1, visible=0)])
    metrics = result["groups"][0]["detection"]
    assert metrics["recall"] == 0
    assert metrics["precision"] is None
    assert metrics["false_positives"] == 0
    assert metrics["mean_matched_iou"] is None


def test_gt_size_bins_do_not_use_predicted_size(tmp_path):
    result = evaluate(tmp_path, [prediction(0, width=40, height=60), prediction(1, width=32, height=32),
                                prediction(2, width=96, height=96)],
                      [annotation(0, width=20, height=30), annotation(1, width=32, height=32),
                       annotation(2, width=96, height=96)])
    sizes = result["groups"][0]["size_bins"]
    assert sizes["small_lt_32_squared"]["recall"] == 0
    assert sizes["medium_32_squared_to_96_squared"]["recall"] == 1
    assert sizes["large_ge_96_squared"]["recall"] == 1


def test_missing_prediction_row_is_fn_and_unknown_duration(tmp_path):
    result = evaluate(tmp_path, [prediction(0), prediction(3)], [annotation(i) for i in range(4)])
    group = result["groups"][0]
    assert group["detection"]["recall"] == 0.5
    assert group["coverage"]["missing_prediction_frames"] == 2
    assert group["coverage"]["missing_prediction_frame_ids"] == [1, 2]
    assert group["tracking"]["longest_lost_run_frames"] == 2
    assert group["tracking"]["runs"][0]["duration_s"] is None
    assert group["tracking"]["reacquisition_time_s"]["count"] == 0


def test_frame_number_gap_ends_loss_interval(tmp_path):
    result = evaluate(tmp_path, [miss(0), prediction(5)], [annotation(0), annotation(5)])
    tracking = result["groups"][0]["tracking"]
    assert tracking["runs"][0]["censored"] is True
    assert tracking["runs"][0]["duration_s"] == 0
    assert tracking["reacquisition_time_s"]["count"] == 0


def test_loss_duration_uses_source_time_not_frame_count_or_latency(tmp_path):
    result = evaluate(tmp_path, [prediction(0, timestamp_s=0), miss(1, timestamp_s=0.2),
                                miss(2, timestamp_s=0.9), prediction(3, timestamp_s=1.4)],
                      [annotation(i) for i in range(4)])
    group = result["groups"][0]
    assert group["tracking"]["reacquisition_time_s"]["p50"] == pytest.approx(1.2)
    assert group["latency"]["inference_ms"]["p95"] == 10


def test_methods_and_sessions_are_separate_and_empty_gt_has_no_accuracy(tmp_path):
    result = evaluate(tmp_path, [prediction(0), miss(0, method="sr"), prediction(0, session_id="s2")],
                      [annotation(0), annotation(0, session_id="unused")])
    groups = {(group["session_id"], group["method"]): group for group in result["groups"]}
    assert groups[("s1", "baseline")]["detection"]["recall"] == 1
    assert groups[("s1", "sr")]["detection"]["recall"] == 0
    assert groups[("s2", "baseline")]["evaluation_status"] == "no_reviewed_annotations"
    assert groups[("s2", "baseline")]["detection"]["recall"] is None
    assert groups[("s2", "baseline")]["detection"]["precision"] is None
    assert result["annotation_summary"]["sessions_without_predictions"] == ["unused"]


def test_all_invisible_has_null_recall_but_defined_false_positive_precision(tmp_path):
    result = evaluate(tmp_path, [prediction(0)], [annotation(0, visible=0)])
    assert result["groups"][0]["detection"]["recall"] is None
    assert result["groups"][0]["detection"]["precision"] == 0


@pytest.mark.parametrize("changes", [
    {"frame_id": -1}, {"frame_id": "1.5"}, {"visible": 2}, {"reviewed": "true"},
    {"x": -1}, {"width": 0}, {"height": "NaN"}, {"x": "inf"}, {"y": ""},
])
def test_invalid_annotation_values_are_rejected(tmp_path, changes):
    path = write_csv(tmp_path / "gt.csv", ANNOTATION_FIELDS, [annotation(0, **changes)])
    with pytest.raises(ValueError):
        load_annotations(path)


@pytest.mark.parametrize("changes", [
    {"confidence": 1.1}, {"confidence": -0.1}, {"confidence": ""},
    {"timestamp_s": "nan"}, {"timestamp_s": -1}, {"inference_ms": -1},
    {"observed": 2}, {"width": ""}, {"label": ""}, {"frame_id": -1},
])
def test_invalid_prediction_values_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        evaluate(tmp_path, [prediction(0, **changes)], [annotation(0)])


def test_duplicate_rows_changed_identity_and_bad_timestamps_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate annotation"):
        evaluate(tmp_path, [prediction(0)], [annotation(0), annotation(0)])
    with pytest.raises(ValueError, match="duplicate prediction"):
        evaluate(tmp_path, [prediction(0), prediction(0)], [annotation(0)])
    with pytest.raises(ValueError, match="one fixed object_id"):
        evaluate(tmp_path, [prediction(0)], [annotation(0), annotation(1, object_id="different")])
    with pytest.raises(ValueError, match="timestamps must increase"):
        evaluate(tmp_path, [prediction(0), prediction(1, timestamp_s=0)], [annotation(0)])


def test_iou_matches_geometry():
    box = dict(x=0, y=0, width=10, height=10)
    assert box_iou(box, dict(x=5, y=0, width=10, height=10)) == pytest.approx(1 / 3)
    assert box_iou(box, dict(x=20, y=20, width=10, height=10)) == 0
    assert box_iou(box, dict(x=None, y=None, width=None, height=None)) == 0
