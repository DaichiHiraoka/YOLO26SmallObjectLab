from __future__ import annotations

import hashlib
import math
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .metrics import Detection, largest


FSRCNN_X2_URL = (
    "https://raw.githubusercontent.com/Saafke/"
    "FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb"
)
FSRCNN_X2_SHA256 = (
    "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c"
)
LOCAL_SR_PADDING_RATIO = 0.20
SUPPORTED_SR_SCALES = (2, 4, 8)
DEFAULT_SR_TARGET_MIN_PX = 96
DEFAULT_SR_MAX_INPUT_SIDE = 640


@dataclass(frozen=True)
class CropRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SuperResolutionDecision:
    target: Detection | None
    reason: str = ""

    @property
    def should_apply(self) -> bool:
        return self.target is not None


def select_super_resolution_target(
    detections: Iterable[Detection],
) -> Detection | None:
    """Select the target for local enhancement.

    This is intentionally isolated because the selection policy will change.
    The initial policy requested for the experiment is the largest bbox.
    """

    return largest(detections)


def expanded_crop_region(
    detection: Detection,
    frame_width: int,
    frame_height: int,
    *,
    padding_ratio: float = LOCAL_SR_PADDING_RATIO,
) -> CropRegion:
    """Return a local crop with modest context around the target bbox.

    The padding is applied once on each side.  A 20% ratio therefore produces
    a crop that is approximately 1.4 times the bbox width and height, unless
    the crop reaches a frame edge.
    """

    padding_x = max(8, round(detection.width * padding_ratio))
    padding_y = max(8, round(detection.height * padding_ratio))
    x1 = max(0, detection.x - padding_x)
    y1 = max(0, detection.y - padding_y)
    x2 = min(frame_width, detection.x + detection.width + padding_x)
    y2 = min(frame_height, detection.y + detection.height + padding_y)
    return CropRegion(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))


def remap_from_super_resolution(
    detection: Detection,
    region: CropRegion,
    *,
    scale: int,
    frame_width: int,
    frame_height: int,
) -> Detection:
    x = region.x + round(detection.x / scale)
    y = region.y + round(detection.y / scale)
    width = max(1, round(detection.width / scale))
    height = max(1, round(detection.height / scale))
    x = min(max(0, x), max(0, frame_width - 1))
    y = min(max(0, y), max(0, frame_height - 1))
    width = min(width, max(1, frame_width - x))
    height = min(height, max(1, frame_height - y))
    return replace(
        detection,
        x=x,
        y=y,
        width=width,
        height=height,
        enhanced=True,
        enhancement_scale=scale,
    )


def dynamic_super_resolution_scales(
    detection: Detection,
    region: CropRegion,
    *,
    target_min_px: int = DEFAULT_SR_TARGET_MIN_PX,
    max_scale: int = 8,
    max_input_side: int = DEFAULT_SR_MAX_INPUT_SIDE,
) -> tuple[int, ...]:
    """Choose an initial SR scale and the larger fallback scales to try.

    The initial scale aims to make the target's shorter bbox side at least
    ``target_min_px`` before YOLO preprocessing.  The local crop is kept near
    ``max_input_side`` where possible so an unnecessarily large SR image is
    not immediately reduced again by YOLO.  If no supported scale fits, x2 is
    retained as the required minimum whenever SR is active.
    """

    if target_min_px <= 0:
        raise ValueError("target_min_px must be greater than 0")
    if max_scale not in SUPPORTED_SR_SCALES:
        raise ValueError(f"max_scale must be one of {SUPPORTED_SR_SCALES}")
    if max_input_side <= 0:
        raise ValueError("max_input_side must be greater than 0")

    supported = tuple(
        scale for scale in SUPPORTED_SR_SCALES if scale <= max_scale
    )
    short_side = max(1, min(detection.width, detection.height))
    required_scale = max(2, math.ceil(target_min_px / short_side))
    preferred = next(
        (scale for scale in supported if scale >= required_scale),
        supported[-1],
    )

    region_long_side = max(1, region.width, region.height)
    fitting = tuple(
        scale
        for scale in supported
        if region_long_side * scale <= max_input_side
    )
    allowed_max = fitting[-1] if fitting else supported[0]
    initial = min(preferred, allowed_max)
    return tuple(
        scale for scale in supported if initial <= scale <= allowed_max
    )


class LargestObjectInstabilityMonitor:
    def __init__(
        self,
        *,
        max_missed_frames: int = 2,
        confidence_drop_ratio: float = 0.75,
        area_drop_ratio: float = 0.82,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.confidence_drop_ratio = confidence_drop_ratio
        self.area_drop_ratio = area_drop_ratio
        self.previous_target: Detection | None = None
        self.missed_frames = 0

    def reset(self) -> None:
        self.previous_target = None
        self.missed_frames = 0

    def decide(
        self,
        detections: Iterable[Detection],
        *,
        frame_width: int,
        frame_height: int,
        trigger_confidence: float,
        trigger_area_pct: float,
    ) -> SuperResolutionDecision:
        current = select_super_resolution_target(detections)
        previous = self.previous_target

        if current is None:
            if previous is not None and self.missed_frames < self.max_missed_frames:
                self.missed_frames += 1
                return SuperResolutionDecision(previous, "detection_missing")
            self.reset()
            return SuperResolutionDecision(None)

        self.previous_target = current
        self.missed_frames = 0
        frame_area = max(1, frame_width * frame_height)
        area_pct = 100.0 * current.area_px / frame_area

        if current.confidence <= trigger_confidence:
            return SuperResolutionDecision(current, "low_confidence")
        if area_pct <= trigger_area_pct:
            return SuperResolutionDecision(current, "small_bbox")
        if previous is not None and previous.label == current.label:
            if current.confidence < previous.confidence * self.confidence_drop_ratio:
                return SuperResolutionDecision(current, "confidence_drop")
            if current.area_px < previous.area_px * self.area_drop_ratio:
                return SuperResolutionDecision(current, "bbox_shrinking")
        return SuperResolutionDecision(None)


class FSRCNNSuperResolution:
    scale = 2
    max_scale = 8

    def __init__(
        self,
        model_path: Path = Path("models/FSRCNN_x2.pb"),
        *,
        model_url: str = FSRCNN_X2_URL,
        expected_sha256: str = FSRCNN_X2_SHA256,
    ) -> None:
        self.model_path = model_path
        self.model_url = model_url
        self.expected_sha256 = expected_sha256.casefold()
        self._upsampler: Any = None
        self._load_error: RuntimeError | None = None

    @property
    def ready(self) -> bool:
        return self._upsampler is not None

    def enhance(self, image: np.ndarray, *, scale: int = 2) -> np.ndarray:
        if image.size == 0:
            raise ValueError("超解像へ渡す画像が空です")
        if scale not in SUPPORTED_SR_SCALES:
            raise ValueError(
                f"超解像倍率は{SUPPORTED_SR_SCALES}から指定してください"
            )
        upsampler = self._get_upsampler()
        enhanced = image
        current_scale = 1
        while current_scale < scale:
            enhanced = upsampler.upsample(enhanced)
            current_scale *= self.scale
        return enhanced

    def _get_upsampler(self) -> Any:
        if self._upsampler is not None:
            return self._upsampler
        if self._load_error is not None:
            raise self._load_error
        try:
            model_path = self._ensure_model_file()
            module = getattr(cv2, "dnn_superres", None)
            if module is None:
                raise RuntimeError(
                    "OpenCVのdnn_superresが利用できません。"
                    "uv syncを再実行してください。"
                )
            upsampler = module.DnnSuperResImpl_create()
            upsampler.readModel(str(model_path))
            upsampler.setModel("fsrcnn", self.scale)
            self._upsampler = upsampler
            return upsampler
        except Exception as error:
            self._load_error = RuntimeError(f"FSRCNNを準備できません: {error}")
            raise self._load_error from error

    def _ensure_model_file(self) -> Path:
        if self.model_path.exists():
            self._verify_model(self.model_path)
            return self.model_path

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.model_path.with_suffix(".pb.download")
        request = urllib.request.Request(
            self.model_url,
            headers={"User-Agent": "YOLO26SmallObjectLab/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            temporary_path.write_bytes(payload)
            self._verify_model(temporary_path)
            temporary_path.replace(self.model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return self.model_path

    def _verify_model(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest().casefold()
        if digest != self.expected_sha256:
            raise RuntimeError(
                "FSRCNNモデルのSHA-256が一致しません。"
                f" expected={self.expected_sha256} actual={digest}"
            )
