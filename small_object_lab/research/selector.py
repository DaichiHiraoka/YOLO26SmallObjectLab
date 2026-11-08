"""Offline, supervised action selection using reviewed boxes, never confidence labels.

One small ridge regressor estimates utility for each recorded action. Utility is
``correct detection - latency_penalty * additional_milliseconds``; correctness
requires a visible, reviewed, same-class ground-truth box with IoU >= 0.5.
Absent actions are not treated as failures. Only the train sessions fit either
the regressors or the feature scaler. Test rows are not scored or transformed.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .evaluation import box_iou, load_annotations


FEATURE_NAMES = (
    "width_frac", "height_frac", "confidence", "speed_norm", "missed_s",
    "area_change", "confidence_change",
)
MODEL_FORMAT = "yolo26-action-selector-v1"
RIDGE_ALPHA = 1.0
FEATURE_CLIP = 20.0


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _feature_vector(features: dict) -> np.ndarray:
    if not isinstance(features, dict):
        raise ValueError("features must be an object")
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    return np.asarray([_finite(features[name], name) for name in FEATURE_NAMES])


def _read_json(path: Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish without overwriting an earlier research result.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_split(path: Path) -> dict[str, list[str]]:
    split = _read_json(path)
    if not isinstance(split, dict) or set(split) != {"train", "validation", "test"}:
        raise ValueError("split must contain exactly train, validation and test")
    seen = set()
    for name, sessions in split.items():
        if not isinstance(sessions, list):
            raise ValueError(f"split {name} must be a list of session IDs")
        for session in sessions:
            if not isinstance(session, str) or not session.strip():
                raise ValueError("session IDs must be nonempty strings")
            if session in seen:
                raise ValueError(f"Duplicate/overlapping session in split: {session}")
            seen.add(session)
    if not split["train"]:
        raise ValueError("At least one train session is required")
    return split


def _load_samples(path: Path, split: dict) -> tuple[dict, int]:
    assignments = {sid: group for group, ids in split.items() for sid in ids}
    samples = {"train": [], "validation": []}
    test_count = 0
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Sample line {line_number} must be an object")
            session = row.get("session_id")
            frame = row.get("frame_id")
            if not isinstance(session, str) or session not in assignments:
                raise ValueError(f"Unassigned sample session at line {line_number}: {session}")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ValueError(f"Invalid frame_id at line {line_number}")
            key = (session, frame)
            if key in seen:
                raise ValueError(f"Duplicate sample frame: {key}")
            seen.add(key)
            group = assignments[session]
            if group == "test":
                # Do not inspect held-out features, candidate outcomes or labels.
                test_count += 1
                continue
            samples[group].append(row)
    return samples, test_count


def _candidate_outcome(candidate: dict, annotation: dict, penalty: float) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("Each candidate must be an object")
    action = candidate.get("action_id")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("Each candidate requires a nonempty action_id")
    if candidate.get("kind") not in {"none", "sr", "bicubic", "crop"}:
        raise ValueError(f"Invalid candidate kind for {action}")
    elapsed = _finite(candidate.get("elapsed_ms"), "elapsed_ms")
    if elapsed < 0:
        raise ValueError("elapsed_ms must be nonnegative")
    detection = candidate.get("detection")
    matched = False
    if detection is not None:
        if not isinstance(detection, dict):
            raise ValueError(f"Invalid detection for {action}")
        for name in ("x", "y", "width", "height", "confidence"):
            value = _finite(detection.get(name), f"detection.{name}")
            if name in {"width", "height"} and value <= 0:
                raise ValueError("Detection dimensions must be positive")
            if name == "confidence" and not 0 <= value <= 1:
                raise ValueError("Detection confidence must be in [0, 1]")
        if not isinstance(detection.get("label"), str):
            raise ValueError("Detection label must be a string")
        matched = bool(
            annotation["visible"]
            and detection["label"] == annotation["label"]
            and box_iou(detection, annotation) >= 0.5
        )
    return {
        "action_id": action, "matched": matched, "elapsed_ms": elapsed,
        "utility": float(matched) - penalty * elapsed,
    }


def _prepare(rows: list, annotations: dict, penalty: float) -> tuple[list, int]:
    prepared = []
    skipped = 0
    for row in rows:
        key = (row["session_id"], row["frame_id"])
        annotation = annotations.get(key)
        if annotation is None or not annotation["reviewed"]:
            skipped += 1
            continue
        vector = _feature_vector(row.get("features"))
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Missing candidate outcomes for {key}")
        outcomes = {}
        for candidate in candidates:
            result = _candidate_outcome(candidate, annotation, penalty)
            action = result["action_id"]
            if action in outcomes:
                raise ValueError(f"Duplicate candidate action {action} for {key}")
            outcomes[action] = result
        if "none" not in outcomes:
            raise ValueError(f"Baseline 'none' candidate is required for {key}")
        prepared.append({
            "session_id": row["session_id"], "features": row["features"],
            "vector": vector, "outcomes": outcomes, "visible": annotation["visible"],
        })
    return prepared, skipped


class LearnedSelector:
    """A transparent JSON-serializable action-value model, without pickle."""

    def __init__(self, payload: dict):
        if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
            raise ValueError("Unsupported selector model format")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Model feature schema does not match this selector")
        sessions = payload.get("training_session_ids")
        if (not isinstance(sessions, list) or not sessions
                or any(not isinstance(sid, str) or not sid.strip() for sid in sessions)
                or len(set(sessions)) != len(sessions)):
            raise ValueError("Invalid model training_session_ids")
        self.training_session_ids = frozenset(sessions)
        self.mean = self._array(payload.get("feature_mean"), len(FEATURE_NAMES))
        self.scale = self._array(payload.get("feature_scale"), len(FEATURE_NAMES))
        if np.any(self.scale <= 0):
            raise ValueError("Model feature scales must be positive")
        actions = payload.get("actions")
        if not isinstance(actions, dict) or "none" not in actions:
            raise ValueError("Model requires a trained 'none' baseline")
        self.coefficients = {}
        for action, values in actions.items():
            if not isinstance(action, str) or not action.strip() or not isinstance(values, dict):
                raise ValueError("Invalid model action")
            self.coefficients[action] = self._array(
                values.get("coefficients"), len(FEATURE_NAMES) + 1,
            )
        self.payload = payload

    @staticmethod
    def _array(values, length: int) -> np.ndarray:
        if not isinstance(values, list) or len(values) != length:
            raise ValueError("Invalid selector model array dimensions")
        return np.asarray([_finite(value, "model array") for value in values])

    @classmethod
    def load(cls, path: Path) -> "LearnedSelector":
        return cls(_read_json(path))

    def choose(self, features: dict, available_actions: list[str]) -> str:
        if (not isinstance(available_actions, list) or not available_actions
                or any(not isinstance(action, str) for action in available_actions)):
            raise ValueError("available_actions must be a nonempty list of action IDs")
        vector = _feature_vector(features)
        vector = np.clip((vector - self.mean) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        design = np.concatenate(([1.0], vector))
        known = set(available_actions).intersection(self.coefficients)
        if not known:
            raise ValueError("No available action was observed in training")
        # Deterministic ties prefer the zero-additional-cost baseline.
        ordered = sorted(known, key=lambda action: (action != "none", action))
        return max(ordered, key=lambda action: float(self.coefficients[action] @ design))


def _validation_report(selector: LearnedSelector, rows: list) -> dict:
    if not rows:
        return {"sample_count": 0, "status": "no_reviewed_validation_samples"}
    matches = visible = 0
    reward = oracle_reward = elapsed = 0.0
    choices = {}
    for row in rows:
        outcomes = row["outcomes"]
        chosen = selector.choose(row["features"], list(outcomes))
        result = outcomes[chosen]
        matches += int(result["matched"])
        visible += int(row["visible"])
        reward += result["utility"]
        oracle_reward += max(item["utility"] for item in outcomes.values())
        elapsed += result["elapsed_ms"]
        choices[chosen] = choices.get(chosen, 0) + 1
    count = len(rows)
    return {
        "sample_count": count, "visible_count": visible, "matched_count": matches,
        "visible_match_rate": matches / visible if visible else None,
        "mean_additional_ms": elapsed / count, "mean_utility": reward / count,
        "oracle_mean_utility": oracle_reward / count,
        "mean_oracle_regret": (oracle_reward - reward) / count,
        "selected_actions": choices,
    }


def train_selector(
    samples_path: Path, annotations_path: Path, split_path: Path, output_path: Path,
    *, latency_penalty: float = 0.002,
) -> dict:
    """Fit on reviewed train frames; report validation only; leave test untouched."""
    penalty = _finite(latency_penalty, "latency_penalty")
    if penalty < 0:
        raise ValueError("latency_penalty must be nonnegative")
    split = _load_split(split_path)
    samples, test_count = _load_samples(samples_path, split)
    configurations = {json.dumps(row["experiment_config"], sort_keys=True)
                      for group in ("train", "validation") for row in samples[group]
                      if "experiment_config" in row}
    if len(configurations) > 1:
        raise ValueError("Candidate samples use different experiment configurations")
    annotations = load_annotations(annotations_path)
    train, train_skipped = _prepare(samples["train"], annotations, penalty)
    validation, validation_skipped = _prepare(samples["validation"], annotations, penalty)
    if not train:
        raise ValueError("No reviewed train annotations match the candidate samples")
    raw = np.vstack([row["vector"] for row in train])
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = np.clip((raw - mean) / scale, -FEATURE_CLIP, FEATURE_CLIP)
    design = np.column_stack((np.ones(len(train)), standardized))
    actions = sorted({action for row in train for action in row["outcomes"]})
    fitted = {}
    regularizer = np.eye(design.shape[1]) * RIDGE_ALPHA
    regularizer[0, 0] = 0.0
    for action in actions:
        indices = [i for i, row in enumerate(train) if action in row["outcomes"]]
        action_design = design[indices]
        targets = np.asarray([train[i]["outcomes"][action]["utility"] for i in indices])
        coefficients = np.linalg.solve(
            action_design.T @ action_design + regularizer, action_design.T @ targets,
        )
        fitted[action] = {"coefficients": coefficients.tolist(), "sample_count": len(indices)}
    payload = {
        "format": MODEL_FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES), "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(), "ridge_alpha": RIDGE_ALPHA,
        "feature_clip": FEATURE_CLIP, "latency_penalty": penalty,
        "objective": "visible_same_label_iou_ge_0.5_minus_additional_ms_penalty",
        "training_session_ids": sorted({row["session_id"] for row in train}),
        "split": split, "actions": fitted,
        "experiment_config": json.loads(next(iter(configurations))) if configurations else None,
        "input_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for name, path in (("samples", samples_path), ("annotations", annotations_path), ("split", split_path))},
    }
    selector = LearnedSelector(payload)
    report = {
        "model_path": str(Path(output_path)), "train_sample_count": len(train),
        "training_session_ids": list(sorted(selector.training_session_ids)),
        "train_skipped_unreviewed_or_missing": train_skipped,
        "validation_skipped_unreviewed_or_missing": validation_skipped,
        "test_samples_not_used": test_count,
        "latency_penalty": penalty,
        "action_sample_counts": {key: value["sample_count"] for key, value in fitted.items()},
        "validation": _validation_report(selector, validation),
    }
    payload["training_report"] = report
    _atomic_json(output_path, payload)
    return report
