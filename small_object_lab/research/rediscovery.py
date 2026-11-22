"""Re-discovery of a lost target by appearance, without extra inference.

When the tracker has no target left to crop around, the detector has often still
put a box on the object: at low confidence, or under another class (a black cup
seen from the front comes out as "cell phone"). The same full-frame inference is
therefore read at a lower threshold and for every class, and each box is compared
with crops of the target that were confirmed earlier in the session. A box is
re-adopted as the target only when it looks like those crops.

Appearance is a colour histogram (HSV, 8x4x4 bins, Bhattacharyya coefficient)
plus a coarse grey-level shape (16x24, normalised cross-correlation), each worth
half. Boxes whose area differs from the reference by 3x or more are not compared.
The defaults were chosen on development recordings only; see docs/RESEARCH_WORKFLOW.md.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..metrics import Detection


@dataclass(frozen=True)
class RediscoveryConfig:
    threshold: float = 0.75
    target_confidence: float = 0.01   # hypotheses of the target class
    other_confidence: float = 0.10    # hypotheses of any other class
    gallery_size: int = 20
    gallery_every: int = 5            # keep every Nth confirmed detection
    max_area_ratio: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if not 0 < self.target_confidence <= 1 or not 0 < self.other_confidence <= 1:
            raise ValueError("hypothesis confidences must be in (0, 1]")
        if self.gallery_size < 1 or self.gallery_every < 1 or self.max_area_ratio <= 1:
            raise ValueError("gallery_size and gallery_every must be >= 1, max_area_ratio > 1")


@dataclass(frozen=True)
class Appearance:
    histogram: np.ndarray
    shape: np.ndarray
    area: float


def appearance(image: np.ndarray, box: Detection) -> Appearance | None:
    x0, y0 = max(0, int(box.x)), max(0, int(box.y))
    x1, y1 = min(image.shape[1], int(box.x + box.width)), min(image.shape[0], int(box.y + box.height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    histogram /= histogram.sum() + 1e-9
    grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 24), interpolation=cv2.INTER_AREA)
    grey = grey.astype(np.float32).flatten()
    grey = (grey - grey.mean()) / (grey.std() + 1e-6)
    return Appearance(histogram, grey, float((x1 - x0) * (y1 - y0)))


def similarity(a: Appearance, b: Appearance, max_area_ratio: float = 3.0) -> float:
    """0..1; -1 when the sizes are too different to be the same object."""
    if abs(math.log((a.area + 1) / (b.area + 1))) >= math.log(max_area_ratio):
        return -1.0
    colour = float(np.sum(np.sqrt(a.histogram * b.histogram)))
    shape = float(np.mean(a.shape * b.shape))
    return 0.5 * colour + 0.5 * max(shape, 0.0)


def split_detections(raw: list[Detection], label: str, confidence: float,
                     config: RediscoveryConfig) -> tuple[list[Detection], list[Detection]]:
    """One low-threshold, all-class inference -> (normal target detections, hypotheses)."""
    target = label.casefold()
    base, hypotheses = [], []
    for item in raw:
        own = item.label.casefold() == target
        if own and item.confidence >= confidence:
            base.append(item)
        elif item.confidence >= (config.target_confidence if own else config.other_confidence):
            hypotheses.append(item)
    return base, hypotheses


class Gallery:
    """Crops of the target that the method itself confirmed (never re-discovered ones)."""

    def __init__(self, config: RediscoveryConfig) -> None:
        self.config = config
        self.items: deque[Appearance] = deque(maxlen=config.gallery_size)
        self.confirmed = 0

    def add(self, image: np.ndarray, box: Detection) -> None:
        if self.confirmed % self.config.gallery_every == 0:
            item = appearance(image, box)
            if item is not None:
                self.items.append(item)
        self.confirmed += 1

    def best(self, image: np.ndarray, hypotheses: list[Detection]) -> tuple[Detection | None, float]:
        best_box, best_score = None, -1.0
        if not self.items:
            return None, best_score
        for hypothesis in hypotheses:
            item = appearance(image, hypothesis)
            if item is None:
                continue
            score = max(similarity(item, ref, self.config.max_area_ratio) for ref in self.items)
            if score > best_score:
                best_box, best_score = hypothesis, score
        return best_box, best_score
