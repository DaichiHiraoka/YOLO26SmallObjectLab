"""A common, explicitly single-target motion tracker for controlled experiments.

This is not a general multi-object tracker or a claim of ByteTrack equivalence.
Only observed boxes count as detections; predictions only propose recovery ROIs.
"""
from __future__ import annotations

import math
from dataclasses import replace

from ..metrics import Detection, coverage, intersection_over_union, largest


class TargetTracker:
    def __init__(self, retention_s: float = 0.75) -> None:
        if not math.isfinite(retention_s) or retention_s <= 0:
            raise ValueError("retention_s must be positive and finite")
        self.retention_s = retention_s
        self.last: Detection | None = None
        self.last_time: float | None = None
        self.track_id = 0
        self.velocity = (0.0, 0.0)
        self.area_change = 0.0
        self.confidence_change = 0.0

    def missed_s(self, timestamp_s: float) -> float:
        return 0.0 if self.last_time is None else max(0.0, timestamp_s - self.last_time)

    def predict(self, timestamp_s: float) -> Detection | None:
        if self.last is None or self.missed_s(timestamp_s) > self.retention_s:
            return None
        dt = self.missed_s(timestamp_s)
        return replace(self.last, x=round(self.last.x + self.velocity[0] * dt),
                       y=round(self.last.y + self.velocity[1] * dt))

    def match(self, detections: list[Detection], timestamp_s: float) -> Detection | None:
        predicted = self.predict(timestamp_s)
        if predicted is None:
            return largest(detections)
        matches = []
        for item in detections:
            if item.label != predicted.label:
                continue
            iou = intersection_over_union(predicted, item)
            dx = item.x + item.width / 2 - predicted.x - predicted.width / 2
            dy = item.y + item.height / 2 - predicted.y - predicted.height / 2
            distance = math.hypot(dx, dy) / max(1, math.hypot(predicted.width, predicted.height))
            # A detection enclosing most of the predicted box is the same object
            # seen whole, even when the tracked box had shrunk to one of its parts.
            if iou >= 0.1 or distance <= 0.75 or coverage(predicted, item) >= 0.5:
                matches.append((iou - 0.2 * distance, item.confidence, item))
        return max(matches, key=lambda x: x[:2])[2] if matches else None

    def update(self, observed: Detection | None, timestamp_s: float) -> None:
        if observed is None:
            return
        previous = self.last
        if previous is None or self.predict(timestamp_s) is None:
            self.track_id += 1
            self.velocity = (0.0, 0.0)
            self.area_change = self.confidence_change = 0.0
        else:
            dt = timestamp_s - self.last_time
            if dt > 0:
                vx = ((observed.x + observed.width / 2) - (previous.x + previous.width / 2)) / dt
                vy = ((observed.y + observed.height / 2) - (previous.y + previous.height / 2)) / dt
                self.velocity = (0.5 * self.velocity[0] + 0.5 * vx,
                                 0.5 * self.velocity[1] + 0.5 * vy)
            self.area_change = observed.area_px / max(1, previous.area_px) - 1
            self.confidence_change = observed.confidence - previous.confidence
        self.last, self.last_time = observed, timestamp_s

    def features(self, target: Detection, timestamp_s: float, width: int, height: int,
                 current: bool) -> dict[str, float]:
        return {
            "width_frac": target.width / width,
            "height_frac": target.height / height,
            "confidence": target.confidence,
            "speed_norm": math.hypot(*self.velocity) / max(1, math.hypot(width, height)),
            "missed_s": 0.0 if current else self.missed_s(timestamp_s),
            "area_change": (target.area_px / max(1, self.last.area_px) - 1
                            if current and self.last else self.area_change),
            "confidence_change": (target.confidence - self.last.confidence
                                  if current and self.last else self.confidence_change),
        }
