"""Ground-truth evaluation of one target per video, without scoring unknown labels.

Coordinates and size bins are in the original video pixels, not resized inputs.
Each prediction method must emit one row per source frame, including misses.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANNOTATION_FIELDS = (
    "session_id", "frame_id", "object_id", "label", "visible",
    "x", "y", "width", "height", "reviewed",
)
PREDICTION_FIELDS = (
    "session_id", "method", "frame_id", "timestamp_s", "track_id", "label",
    "confidence", "x", "y", "width", "height", "observed", "inference_ms",
    "action_id",
)
BOX_FIELDS = ("x", "y", "width", "height")


def _rows(path: Path, required: tuple[str, ...]):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            location = f"{path}:{line}"
            if None in row or any(row[name] is None for name in required):
                raise ValueError(f"{location}: wrong number of CSV values")
            yield location, {key: value.strip() for key, value in row.items()}


def _text(row: dict, key: str, location: str) -> str:
    value = row[key]
    if not value:
        raise ValueError(f"{location}: {key} must not be empty")
    return value


def _frame_id(row: dict, location: str) -> int:
    value = row["frame_id"]
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{location}: frame_id must be a nonnegative integer")
    return int(value)


def _flag(row: dict, key: str, location: str) -> bool:
    if row[key] not in {"0", "1"}:
        raise ValueError(f"{location}: {key} must be 0 or 1")
    return row[key] == "1"


def _number(row: dict, key: str, location: str, *, positive: bool = False) -> float:
    try:
        value = float(row[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: {key} must be a finite number") from error
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{location}: {key} must be finite and {relation}")
    return value


def _box(row: dict, location: str, *, required: bool) -> dict:
    provided = [bool(row[key]) for key in BOX_FIELDS]
    if not any(provided):
        if required:
            raise ValueError(f"{location}: this row requires a bounding box")
        return {key: None for key in BOX_FIELDS}
    if not all(provided):
        raise ValueError(f"{location}: provide all four bbox values or leave all empty")
    return {
        key: _number(row, key, location, positive=key in {"width", "height"})
        for key in BOX_FIELDS
    }


def load_annotations(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read validated annotations; unreviewed rows may have no box.

    One fixed object_id and label are allowed per session. Multiple sessions
    may annotate different objects. Off-image bounds require video dimensions
    to check and are deliberately not inferred from the supplied CSV alone.
    """
    result = {}
    identities = {}
    for location, row in _rows(path, ANNOTATION_FIELDS):
        session = _text(row, "session_id", location)
        frame = _frame_id(row, location)
        object_id = _text(row, "object_id", location)
        label = _text(row, "label", location)
        reviewed = _flag(row, "reviewed", location)
        visible = _flag(row, "visible", location) if reviewed or row["visible"] else False
        key = (session, frame)
        if key in result:
            raise ValueError(f"{location}: duplicate annotation for {key}")
        identity = (object_id, label)
        if session in identities and identities[session] != identity:
            raise ValueError(f"{location}: only one fixed object_id and label per session")
        identities[session] = identity
        result[key] = {
            "session_id": session, "frame_id": frame, "object_id": object_id,
            "label": label, "visible": visible, "reviewed": reviewed,
            **_box(row, location, required=reviewed and visible),
        }
    return result


def _load_predictions(path: Path) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    groups = defaultdict(dict)
    for location, row in _rows(path, PREDICTION_FIELDS):
        session = _text(row, "session_id", location)
        method = _text(row, "method", location)
        frame = _frame_id(row, location)
        observed = _flag(row, "observed", location)
        box = _box(row, location, required=observed)
        if observed:
            _text(row, "label", location)
        confidence = None
        if row["confidence"]:
            confidence = _number(row, "confidence", location)
            if confidence > 1:
                raise ValueError(f"{location}: confidence must be between 0 and 1")
        elif observed:
            raise ValueError(f"{location}: observed detection requires confidence")
        key = (session, method)
        if frame in groups[key]:
            raise ValueError(f"{location}: duplicate prediction for {key}, frame {frame}")
        groups[key][frame] = {
            "session_id": session, "method": method, "frame_id": frame,
            "timestamp_s": _number(row, "timestamp_s", location),
            "track_id": row["track_id"], "label": row["label"],
            "confidence": confidence, "observed": observed,
            "inference_ms": _number(row, "inference_ms", location),
            "action_id": row["action_id"], **box,
        }
    for key, predictions in groups.items():
        ordered = [predictions[frame] for frame in sorted(predictions)]
        times = [item["timestamp_s"] for item in ordered]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError(f"{path}: timestamps must increase with frame_id for {key}")
    return dict(groups)


def box_iou(left: dict, right: dict) -> float:
    """IoU for validated x/y/width/height boxes; a missing box has zero IoU."""
    if any(box.get(key) is None for box in (left, right) for key in BOX_FIELDS):
        return 0.0
    overlap_width = max(0.0, min(left["x"] + left["width"], right["x"] + right["width"])
                        - max(left["x"], right["x"]))
    overlap_height = max(0.0, min(left["y"] + left["height"], right["y"] + right["height"])
                         - max(left["y"], right["y"]))
    intersection = overlap_width * overlap_height
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _timing(values: list[float]) -> dict:
    return {
        "count": len(values), "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95), "max": max(values) if values else None,
    }


def _size_bin(annotation: dict) -> str:
    area = annotation["width"] * annotation["height"]
    if area < 32 * 32:
        return "small_lt_32_squared"
    if area < 96 * 96:
        return "medium_32_squared_to_96_squared"
    return "large_ge_96_squared"


def _loss_runs(events: list[dict]) -> list[dict]:
    """Loss intervals use source timestamps, not frame count / measured FPS.

    Unknown or invisible annotations end an interval. A run with no following
    recovery has only a lower-bound duration, ending at its final missed frame.
    Missing prediction timestamps make duration unknown, never interpolated.
    """
    runs = []
    active = []

    def finish(recovery=None):
        if not active:
            return
        times = [event["timestamp_s"] for event in active]
        end_time = recovery["timestamp_s"] if recovery is not None else times[-1]
        duration = None
        if all(time is not None for time in times) and end_time is not None:
            duration = end_time - times[0]
        runs.append({
            "start_frame": active[0]["frame_id"], "last_missed_frame": active[-1]["frame_id"],
            "missed_frames": len(active), "start_timestamp_s": times[0],
            "recovered_frame": recovery["frame_id"] if recovery is not None else None,
            "duration_s": duration, "censored": recovery is None,
            "duration_is_lower_bound": recovery is None,
            "timestamps_complete": all(time is not None for time in times) and end_time is not None,
        })
        active.clear()

    previous_frame = None
    for event in events:
        if previous_frame is not None and event["frame_id"] != previous_frame + 1:
            finish()
        previous_frame = event["frame_id"]
        if not event["eligible"]:
            finish()
        elif event["matched"]:
            finish(event)
        else:
            active.append(event)
    finish()
    return runs


def _evaluate_group(session: str, method: str, predictions: dict, annotations: dict,
                    iou_threshold: float) -> dict:
    reviewed = {frame: row for frame, row in annotations.items() if row["reviewed"]}
    visible = {frame: row for frame, row in reviewed.items() if row["visible"]}
    tp = fp = fn = true_negative = 0
    matched_ious = []
    visible_ious = []
    matched_ids = []
    events = []
    sizes = {name: {"visible_frames": 0, "true_positives": 0} for name in (
        "small_lt_32_squared", "medium_32_squared_to_96_squared", "large_ge_96_squared",
    )}
    for frame in sorted(set(predictions) | set(annotations)):
        prediction = predictions.get(frame)
        annotation = annotations.get(frame)
        eligible = bool(annotation and annotation["reviewed"] and annotation["visible"])
        observed = bool(prediction and prediction["observed"])
        iou = 0.0
        same_label = bool(prediction and annotation and prediction["label"] == annotation["label"])
        if eligible and observed and same_label:
            iou = box_iou(prediction, annotation)
        matched = eligible and observed and same_label and iou >= iou_threshold
        events.append({
            "frame_id": frame, "timestamp_s": prediction["timestamp_s"] if prediction else None,
            "eligible": eligible, "matched": matched,
        })
        if annotation is None or not annotation["reviewed"]:
            continue
        if annotation["visible"]:
            size = sizes[_size_bin(annotation)]
            size["visible_frames"] += 1
            visible_ious.append(iou)
            if matched:
                tp += 1
                size["true_positives"] += 1
                matched_ious.append(iou)
                matched_ids.append(prediction["track_id"])
            else:
                fn += 1
                fp += int(observed)
        elif observed:
            fp += 1
        else:
            true_negative += 1
    for values in sizes.values():
        values["recall"] = _ratio(values["true_positives"], values["visible_frames"])

    id_counts = Counter(track_id for track_id in matched_ids if track_id)
    known_ids = [track_id for track_id in matched_ids if track_id]
    dominant_count = max(id_counts.values(), default=0)
    switches = sum(left != right for left, right in zip(known_ids, known_ids[1:]))
    losses = _loss_runs(events)
    recovered_durations = [run["duration_s"] for run in losses
                           if not run["censored"] and run["duration_s"] is not None]
    loss_durations = [run["duration_s"] for run in losses if run["duration_s"] is not None]
    prediction_times = [predictions[frame]["timestamp_s"] for frame in sorted(predictions)]
    duration = prediction_times[-1] - prediction_times[0] if len(prediction_times) > 1 else 0.0
    recall, precision = _ratio(tp, tp + fn), _ratio(tp, tp + fp)
    f1 = None
    if recall is not None and precision is not None:
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    missing_frames = sorted(set(reviewed) - set(predictions))
    return {
        "session_id": session, "method": method,
        "evaluation_status": "evaluated" if reviewed else "no_reviewed_annotations",
        "coverage": {
            "prediction_frames": len(predictions), "annotation_frames": len(annotations),
            "reviewed_frames": len(reviewed), "reviewed_visible_frames": len(visible),
            "reviewed_invisible_frames": len(reviewed) - len(visible),
            "reviewed_prediction_frames": len(set(predictions) & set(reviewed)),
            "reviewed_fraction_of_predictions": _ratio(len(set(predictions) & set(reviewed)), len(predictions)),
            "prediction_frames_without_annotation": len(set(predictions) - set(annotations)),
            "prediction_frames_unreviewed": sum(frame in annotations and not annotations[frame]["reviewed"]
                                                for frame in predictions),
            "missing_prediction_frames": len(missing_frames),
            "missing_prediction_frame_ids": missing_frames,
            "missing_visible_prediction_frames": len(set(visible) - set(predictions)),
        },
        "detection": {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_negative_frames": true_negative,
            "recall": recall, "precision": precision, "f1": f1,
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
            "mean_visible_iou_including_misses": sum(visible_ious) / len(visible_ious) if visible_ious else None,
        },
        "tracking": {
            "id_switches": switches if known_ids else None,
            "matched_frames_with_id": len(known_ids),
            "matched_id_coverage": _ratio(len(known_ids), tp),
            "distinct_matched_ids": len(id_counts),
            "dominant_id": id_counts.most_common(1)[0][0] if id_counts else None,
            "dominant_id_fraction_of_matches": _ratio(dominant_count, tp) if known_ids else None,
            "dominant_id_visible_recall": _ratio(dominant_count, len(visible)) if known_ids else None,
            "lost_runs": len(losses), "lost_frames": fn,
            "longest_lost_run_frames": max((run["missed_frames"] for run in losses), default=0),
            "lost_duration_s": _timing(loss_durations),
            "reacquisition_time_s": _timing(recovered_durations),
            "censored_runs": sum(run["censored"] for run in losses),
            "runs_with_unknown_duration": sum(run["duration_s"] is None for run in losses),
            "runs": losses,
        },
        "latency": {
            "inference_ms": _timing([row["inference_ms"] for row in predictions.values()]),
            "source_duration_s": duration,
            "source_sample_rate_fps": (len(predictions) - 1) / duration if duration > 0 else None,
        },
        "size_bins": sizes,
    }


def evaluate_predictions(predictions_path: Path, annotations_path: Path, *,
                         iou_threshold: float = 0.5) -> dict:
    """Evaluate each session/method separately and return JSON-safe results.

    This evaluates a single target per session, not all same-class instances.
    It does not compute mAP, MOTA, HOTA or IDF1. A wrong-label / low-IoU observed
    box is both FP and FN on visible frames. Motion-only boxes are not observed
    detections. Known visible frames with absent prediction rows are FN, with
    absent timestamps explicitly marked unknown in the loss intervals.
    """
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    annotations = load_annotations(annotations_path)
    predictions = _load_predictions(predictions_path)
    by_session = defaultdict(dict)
    for (session, frame), annotation in annotations.items():
        by_session[session][frame] = annotation
    predicted_sessions = {session for session, _ in predictions}
    return {
        "schema_version": 1, "iou_threshold": iou_threshold,
        "annotation_summary": {
            "frames": len(annotations),
            "reviewed_frames": sum(row["reviewed"] for row in annotations.values()),
            "sessions": sorted(by_session),
            "sessions_without_predictions": sorted(set(by_session) - predicted_sessions),
        },
        "groups": [_evaluate_group(session, method, rows, by_session.get(session, {}), iou_threshold)
                   for (session, method), rows in sorted(predictions.items())],
        "notes": [
            "Only reviewed annotations are scored; unknown and unreviewed frames are excluded.",
            "Missing prediction rows at reviewed visible frames count as false negatives; coverage reports these gaps.",
            "Recall is null with zero visible GT; precision is null with zero scored observed detections; matched IoU is null with zero TP.",
            "Predicted-only (observed=0) boxes do not count as detections or false positives.",
            "ID switches compare successive correct detections with known IDs, including across occlusions or annotation gaps; ID coverage is reported separately.",
            "Loss runs end at unknown/invisible frames or frame-number gaps. Censored durations are lower bounds; missing timestamps yield null.",
            "Reacquisition time starts at the first missed visible frame and ends at the next matched frame in the same fully annotated visible segment.",
            "Size bins use original GT area: small < 32^2, medium 32^2 to < 96^2, large >= 96^2 pixels.",
            "Source sample rate describes video timestamps and is not processing throughput. Inference latency excludes unrecorded I/O.",
            "This single-target evaluation is not mAP, IDF1, MOTA or HOTA.",
        ],
    }
