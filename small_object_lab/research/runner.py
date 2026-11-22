"""Paired offline experiments. Every method receives exactly the same frames.

Base YOLO predictions are cached once per frame. Reported method latency includes
that shared measured base cost plus its own work, not the multi-method wall time.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..metrics import Detection, intersection_over_union, plausible_replacement
from ..super_resolution import (DEFAULT_SR_MODEL, SR_MODEL_NAMES, create_super_resolution,
                                dynamic_super_resolution_scales, expanded_crop_region,
                                remap_from_super_resolution)
from .sources import FrameSource, sha256_file
from .rediscovery import Gallery, RediscoveryConfig, split_detections
from .tracking import TargetTracker, needs_enhancement


@dataclass(frozen=True)
class Action:
    kind: str
    padding: float = 0.0
    scale: int = 1

    @property
    def action_id(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}_p{round(self.padding * 100):03d}_x{self.scale}"


NONE = Action("none")
# Padding per side as a fraction of the box. A key needs surrounding context to be
# recognised, so wide crops (100-300%) are candidates too, not only tight ones.
PADDINGS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0)
SR_ACTIONS = tuple(Action("sr", padding, scale) for padding in PADDINGS for scale in (2, 4, 8))
# x1: re-detect on the crop alone; YOLO's own letterbox resize is the only enlargement.
CROP_ACTIONS = tuple(Action("crop", padding, 1) for padding in PADDINGS)
METHODS = ("baseline", "fixed", "bicubic", "rule", "learned")
# "<method>+rd": the same method plus appearance re-discovery when the target is lost
REDISCOVERY_SUFFIX = "+rd"
ALL_METHODS = METHODS + tuple(method + REDISCOVERY_SUFFIX for method in METHODS)
PREDICTION_FIELDS = ["session_id", "method", "frame_id", "timestamp_s", "track_id", "label",
                     "confidence", "x", "y", "width", "height", "observed", "inference_ms", "action_id"]


@dataclass(frozen=True)
class ExperimentConfig:
    label: str = "cup"
    confidence: float = 0.20
    imgsz: int = 960
    local_imgsz: int = 640
    retention_s: float = 0.75
    trigger_confidence: float = 0.45
    trigger_area_pct: float = 1.0
    fixed_scale: int = 4
    fixed_padding: float = 0.20
    target_min_px: int = 96
    # Engine behind every "sr" action; recorded so selectors and runs stay comparable.
    sr_model: str = DEFAULT_SR_MODEL

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("A target class is required")
        if not 0 < self.confidence <= 1 or not 0 <= self.trigger_confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.fixed_scale not in (2, 4, 8) or self.fixed_padding not in PADDINGS:
            raise ValueError(f"Fixed action must use scale 2/4/8 and padding in {PADDINGS}")
        if any(size < 32 or size % 32 for size in (self.imgsz, self.local_imgsz)):
            raise ValueError("Input dimensions must be positive multiples of 32")
        if not math.isfinite(self.retention_s) or self.retention_s <= 0:
            raise ValueError("retention_s must be positive")
        if not math.isfinite(self.trigger_area_pct) or not 0 <= self.trigger_area_pct <= 100:
            raise ValueError("trigger_area_pct must be between 0 and 100")
        if self.target_min_px <= 0:
            raise ValueError("target_min_px must be positive")
        if self.sr_model not in SR_MODEL_NAMES:
            raise ValueError(f"sr_model must be one of {SR_MODEL_NAMES}")


def detection_dict(detection: Detection | None) -> dict | None:
    if detection is None:
        return None
    return {key: getattr(detection, key) for key in ("label", "confidence", "x", "y", "width", "height")}


class InferenceBackend:
    def __init__(self, model_path: Path, sr_model: str = DEFAULT_SR_MODEL, device: str = "auto",
                 model_dir: Path | None = None) -> None:
        # Local weights only: a misspelled path must not download a different model.
        self.model_path = model_path.resolve(strict=True)
        self.sr_model = sr_model
        self.sr = create_super_resolution(sr_model, model_dir or Path(__file__).resolve().parents[2] / "models")
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = YOLO(str(self.model_path))

    def synchronize(self) -> None:
        if self.device != "cpu" and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def detect(self, image: np.ndarray, config: ExperimentConfig, *, local: bool = False) -> list[Detection]:
        return self._predict(image, config.confidence, config.local_imgsz if local else config.imgsz, config.label)

    def detect_all(self, image: np.ndarray, config: ExperimentConfig, confidence: float) -> list[Detection]:
        """Full-frame boxes of every class down to ``confidence``; one inference, as detect()."""
        return self._predict(image, confidence, config.imgsz, None)

    def _predict(self, image: np.ndarray, confidence: float, imgsz: int, label: str | None) -> list[Detection]:
        result = self.model.predict(image, conf=confidence, imgsz=imgsz, rect=False, device=self.device,
                                    verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        for xyxy, confidence, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                         result.boxes.cls.cpu().numpy()):
            label_name = str(result.names[int(cls)])
            if label is not None and label_name.casefold() != label.casefold():
                continue
            x1, y1, x2, y2 = map(float, xyxy)
            x1, y1 = max(0, round(x1)), max(0, round(y1))
            x2, y2 = min(image.shape[1], round(x2)), min(image.shape[0], round(y2))
            if x2 > x1 and y2 > y1:
                detections.append(Detection(label_name, float(confidence), x1, y1, x2-x1, y2-y1))
        return detections

    def enhance(self, crop: np.ndarray, action: Action) -> np.ndarray:
        if action.kind == "crop":
            return crop
        if action.kind == "sr":
            return self.sr.enhance(crop, scale=action.scale)
        return cv2.resize(crop, None, fx=action.scale, fy=action.scale, interpolation=cv2.INTER_CUBIC)

    def warmup(self, image: np.ndarray, config: ExperimentConfig, *, local: bool) -> None:
        self.detect(image, config)
        if local:
            crop = image[:min(48, image.shape[0]), :min(48, image.shape[1])]
            for scale in (2, 4, 8):
                self.detect(self.enhance(crop, Action("sr", 0.2, scale)), config, local=True)
        self.synchronize()


def available_actions(target: Detection | None, shape: tuple, config: ExperimentConfig) -> list[Action]:
    if target is None:
        return [NONE]
    h, w = shape[:2]
    if target.x >= w or target.y >= h or target.x + target.width <= 0 or target.y + target.height <= 0:
        return [NONE]
    return [NONE] + [action for action in (*SR_ACTIONS, *CROP_ACTIONS) if within_budget(target, w, h, action, config)]


def within_budget(target: Detection, width: int, height: int, action: Action, config: ExperimentConfig) -> bool:
    """Enlarged crops must fit the local input; an unenlarged crop always fits."""
    if action.kind == "crop":
        return True
    region = expanded_crop_region(target, width, height, padding_ratio=action.padding)
    return max(region.width, region.height) * action.scale <= config.local_imgsz


def execute_action(backend: Any, frame: np.ndarray, target: Detection | None,
                   current: Detection | None, action: Action, config: ExperimentConfig) -> dict:
    result = {"action_id": action.action_id, "kind": action.kind, "padding": action.padding,
              "scale": action.scale, "elapsed_ms": 0.0, "detection": detection_dict(current),
              "local_detection": None, "adopted": False, "roi": None}
    if action.kind == "none" or target is None:
        return result
    h, w = frame.shape[:2]
    region = expanded_crop_region(target, w, h, padding_ratio=action.padding)
    if not within_budget(target, w, h, action, config):
        raise ValueError("Requested action exceeds the local input budget")
    backend.synchronize()
    started = time.perf_counter()
    crop = frame[region.y:region.y+region.height, region.x:region.x+region.width]
    enhanced = backend.enhance(crop, action)
    detections = [remap_from_super_resolution(item, region, scale=action.scale,
                                            frame_width=w, frame_height=h)
                  for item in backend.detect(enhanced, config, local=True)]
    matches = [d for d in detections if d.label.casefold() == target.label.casefold()
               and intersection_over_union(d, target) >= 0.05]
    candidate = max(matches, key=lambda d: (intersection_over_union(d, target), d.confidence), default=None)
    backend.synchronize()
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    result["local_detection"] = detection_dict(candidate)
    result["roi"] = asdict(region)
    result["enhanced_width"], result["enhanced_height"] = enhanced.shape[1], enhanced.shape[0]
    # rect=False uses square letterboxing. SR factor does not equal YOLO object magnification.
    gain = config.local_imgsz / max(enhanced.shape[:2])
    result["yolo_gain"] = gain
    result["yolo_target_width_px"] = target.width * action.scale * gain
    result["yolo_target_height_px"] = target.height * action.scale * gain
    reference = current if current is not None else target
    if (candidate is not None
            and plausible_replacement(candidate, reference, reference_observed=current is not None)
            and (current is None or candidate.confidence >= current.confidence)):
        result["detection"] = detection_dict(candidate)
        result["adopted"] = True
    return result


class MethodEngine:
    def __init__(self, method: str, config: ExperimentConfig, selector: Any = None,
                 rediscovery: RediscoveryConfig | None = None) -> None:
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.kind = method.removesuffix(REDISCOVERY_SUFFIX)
        if self.kind == "learned" and selector is None:
            raise ValueError("learned requires a trained selector")
        self.method, self.config, self.selector = method, config, selector
        self.tracker = TargetTracker(config.retention_s)
        self.rediscovery = (rediscovery or RediscoveryConfig()) if method.endswith(REDISCOVERY_SUFFIX) else None
        self.gallery = Gallery(self.rediscovery) if self.rediscovery else None

    def process(self, backend: Any, frame: np.ndarray, base: list[Detection], timestamp_s: float,
                hypotheses: list[Detection] | tuple = ()) -> tuple:
        started = time.perf_counter()
        current = self.tracker.match(base, timestamp_s)
        target = current or self.tracker.predict(timestamp_s)
        features = (self.tracker.features(target, timestamp_s, frame.shape[1], frame.shape[0], current is not None)
                    if target else {})
        feasible = available_actions(target, frame.shape, self.config)
        trigger = bool(target and needs_enhancement(features, missing=current is None,
                                                    trigger_confidence=self.config.trigger_confidence,
                                                    trigger_area_pct=self.config.trigger_area_pct))
        actions = [NONE]
        if trigger and self.kind in {"fixed", "bicubic"}:
            fixed = Action("sr", self.config.fixed_padding, self.config.fixed_scale)
            if fixed in feasible:
                actions = [replace(fixed, kind="bicubic") if self.kind == "bicubic" else fixed]
        elif trigger and self.kind == "rule":
            region = expanded_crop_region(target, frame.shape[1], frame.shape[0], padding_ratio=0.2)
            scales = dynamic_super_resolution_scales(target, region, target_min_px=self.config.target_min_px,
                                                     max_input_side=self.config.local_imgsz)
            actions = [Action("sr", 0.2, scale) for scale in scales if Action("sr", 0.2, scale) in feasible] or [NONE]
        elif trigger and self.kind == "learned":
            choice = self.selector.choose(features, [a.action_id for a in feasible])
            actions = [next(a for a in feasible if a.action_id == choice)]
        attempts = []
        final = current
        for action in actions:
            attempt = execute_action(backend, frame, target, current, action, self.config)
            attempts.append(attempt)
            if attempt["adopted"]:
                final = Detection(**attempt["detection"])
                break
        if self.gallery is not None:
            if final is None and target is None:
                attempts.append(self._rediscover(frame, hypotheses))
                if attempts[-1]["adopted"]:
                    final = Detection(**attempts[-1]["detection"])
            elif final is not None:
                self.gallery.add(frame, final)  # only boxes the method confirmed itself
        self.tracker.update(final, timestamp_s)
        backend.synchronize()
        overhead_ms = (time.perf_counter() - started) * 1000
        return final, features, attempts, overhead_ms


    def _rediscover(self, frame: np.ndarray, hypotheses) -> dict:
        started = time.perf_counter()
        found, score = self.gallery.best(frame, list(hypotheses))
        adopted = found is not None and score >= self.rediscovery.threshold
        detection = (Detection(self.config.label, found.confidence, found.x, found.y, found.width, found.height)
                     if adopted else None)
        return {"action_id": "rediscover", "kind": "rediscover", "padding": 0.0, "scale": 1,
                "elapsed_ms": (time.perf_counter() - started) * 1000, "detection": detection_dict(detection),
                "local_detection": None, "adopted": adopted, "roi": None, "score": score,
                "hypotheses": len(hypotheses), "source_label": found.label if found else None}


def provenance(backend: Any, source: FrameSource, config: ExperimentConfig) -> dict:
    versions = {}
    for package in ("ultralytics", "torch", "numpy", "opencv-python", "opencv-contrib-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code_root = Path(__file__).resolve().parents[2]
    code_hashes = {str(p.relative_to(code_root)).replace("\\", "/"): sha256_file(p)
                   for p in sorted((code_root / "small_object_lab").rglob("*.py"))}
    git = None
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, capture_output=True,
                              text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        git = proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": source.session_id, "source_sha256": source.fingerprint,
            "source_metadata": source.metadata, "config": asdict(config), "git_head": git,
            "code_sha256": code_hashes, "packages": versions,
            "model_sha256": sha256_file(backend.model_path) if hasattr(backend, "model_path") else None,
            "sr_model": getattr(backend, "sr_model", None),
            "sr_sha256": backend.sr.model_sha256() if hasattr(backend, "sr") else None,
            "device": getattr(backend, "device", "test"), "status": "running",
            "tracker": "single_target_iou_velocity_v1",
            "timing": "shared measured base inference + per-method processing; excludes decode, disk I/O and network",
            "warning": "No accuracy claims without reviewed ground truth. Predictor boxes are never counted as detections."}


def run_experiment(source: FrameSource, output: Path, backend: Any, config: ExperimentConfig,
                   *, methods: tuple[str, ...] = ("baseline", "fixed", "bicubic", "rule"),
                   selector: Any = None, max_frames: int | None = None,
                   collect_candidates: bool = False, allow_training_session: bool = False,
                   rediscovery: RediscoveryConfig | None = None) -> dict:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Specify distinct methods")
    rediscovery = rediscovery or RediscoveryConfig()
    engines = [MethodEngine(method, config, selector, rediscovery) for method in methods]
    any_rediscovery = any(engine.rediscovery for engine in engines)
    if selector is not None and source.session_id in set(getattr(selector, "training_session_ids", [])) and not allow_training_session:
        raise ValueError("This session was used to train the selector; reserve it for training, not evaluation")
    if getattr(backend, "sr_model", config.sr_model) != config.sr_model:
        raise ValueError("The backend's super-resolution model differs from config.sr_model")
    trained = selector.payload.get("experiment_config") if selector is not None else None
    # Configurations saved before sr_model was recorded were measured with FSRCNN.
    if trained is not None and {"sr_model": "fsrcnn", **trained} != asdict(config):
        raise ValueError("Selector training configuration differs from this experiment")
    output.mkdir(parents=True, exist_ok=False)
    manifest = provenance(backend, source, config)
    manifest.update(rediscovery=asdict(rediscovery) if any_rediscovery else None)
    manifest.update(methods=list(methods), collect_candidates=collect_candidates,
                    training_session_override=allow_training_session, max_frames=max_frames)
    if selector is not None:
        manifest["selector"] = selector.payload
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_tracker = TargetTracker(config.retention_s)
    count = 0
    latencies = {method: [] for method in methods}
    observed = {method: 0 for method in methods}
    try:
        with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as pred_stream, \
                (output / "attempts.jsonl").open("w", encoding="utf-8") as attempts_stream, \
                (output / "candidates.jsonl").open("w", encoding="utf-8") as candidate_stream:
            writer = csv.DictWriter(pred_stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for input_frame in source:
                if max_frames is not None and count >= max_frames:
                    break
                frame, timestamp_s = input_frame.image, input_frame.timestamp_s
                if count == 0:
                    backend.warmup(frame, config, local=collect_candidates or any(m != "baseline" for m in methods))
                backend.synchronize()
                started = time.perf_counter()
                if any_rediscovery:
                    # one inference at a lower threshold; the normal detections are its subset
                    lowest = min(config.confidence, rediscovery.target_confidence, rediscovery.other_confidence)
                    base, hypotheses = split_detections(backend.detect_all(frame, config, lowest),
                                                        config.label, config.confidence, rediscovery)
                else:
                    base, hypotheses = backend.detect(frame, config), []
                backend.synchronize()
                base_ms = (time.perf_counter() - started) * 1000
                for engine in engines:
                    final, features, attempts, overhead_ms = engine.process(backend, frame, base, timestamp_s,
                                                                            hypotheses)
                    elapsed_ms = base_ms + overhead_ms
                    row = {"session_id": source.session_id, "method": engine.method,
                           "frame_id": input_frame.frame_id, "timestamp_s": timestamp_s,
                           "track_id": engine.tracker.track_id if final else "", "observed": int(final is not None),
                           "inference_ms": elapsed_ms, "action_id": "|".join(a["action_id"] for a in attempts)}
                    row.update(detection_dict(final) or {})
                    writer.writerow(row)
                    attempts_stream.write(json.dumps({"session_id": source.session_id, "method": engine.method,
                        "frame_id": input_frame.frame_id, "timestamp_s": timestamp_s, "features": features,
                        "base_ms": base_ms, "attempts": attempts}, ensure_ascii=False) + "\n")
                    latencies[engine.method].append(elapsed_ms)
                    observed[engine.method] += final is not None
                if collect_candidates:
                    current = candidate_tracker.match(base, timestamp_s)
                    target = current or candidate_tracker.predict(timestamp_s)
                    features = (candidate_tracker.features(target, timestamp_s, frame.shape[1], frame.shape[0], current is not None)
                                if target else dict.fromkeys(("width_frac", "height_frac", "confidence", "speed_norm",
                                                             "missed_s", "area_change", "confidence_change"), 0.0))
                    candidates = [execute_action(backend, frame, target, current, action, config)
                                  for action in available_actions(target, frame.shape, config)]
                    candidate_stream.write(json.dumps({"session_id": source.session_id, "frame_id": input_frame.frame_id,
                        "timestamp_s": timestamp_s, "features": features, "target_available": target is not None,
                        "experiment_config": asdict(config), "source_sha256": source.fingerprint,
                        "candidates": candidates}, ensure_ascii=False) + "\n")
                    # A shared base-only history prevents another candidate's outcome leaking into its features.
                    candidate_tracker.update(current, timestamp_s)
                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} frames", flush=True)
        if count == 0:
            raise ValueError("No frames processed")
        manifest.update(status="complete", processed_frames=count,
                        summary={m: {"observed_frames": observed[m], "latency_p50_ms": float(np.median(v)),
                                     "latency_p95_ms": float(np.percentile(v, 95)),
                                     "compute_fps": 1000 / float(np.mean(v)) if np.mean(v) > 0 else None}
                                 for m, v in latencies.items()})
    except BaseException as error:
        manifest.update(status="failed", processed_frames=count, error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
