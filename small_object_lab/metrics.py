from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    streak: int = 1
    enhanced: bool = False

    @property
    def area_px(self) -> int:
        return self.width * self.height

    def as_dict(self, frame_width: int, frame_height: int) -> dict[str, object]:
        frame_area = max(1, frame_width * frame_height)
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "x": self.x,
            "y": self.y,
            "width_px": self.width,
            "height_px": self.height,
            "area_px": self.area_px,
            "width_pct": round(100.0 * self.width / max(1, frame_width), 4),
            "height_pct": round(100.0 * self.height / max(1, frame_height), 4),
            "area_pct": round(100.0 * self.area_px / frame_area, 6),
            "streak": self.streak,
            "source": "super_resolution" if self.enhanced else "full_frame",
        }


def intersection_over_union(left: Detection, right: Detection) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area_px + right.area_px - intersection
    return intersection / union if union > 0 else 0.0


class StabilityTracker:
    """Adds a consecutive-frame streak to detections without a heavy tracker."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._previous: list[Detection] = []

    def reset(self) -> None:
        self._previous.clear()

    def update(self, detections: Iterable[Detection]) -> list[Detection]:
        updated: list[Detection] = []
        unused_previous = set(range(len(self._previous)))

        for detection in detections:
            best_index = None
            best_iou = self.iou_threshold
            for index in unused_previous:
                previous = self._previous[index]
                if previous.label != detection.label:
                    continue
                overlap = intersection_over_union(detection, previous)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index

            streak = 1
            if best_index is not None:
                streak = self._previous[best_index].streak + 1
                unused_previous.remove(best_index)
            updated.append(replace(detection, streak=streak))

        self._previous = updated
        return updated


def smallest(detections: Iterable[Detection]) -> Detection | None:
    return min(detections, key=lambda item: item.area_px, default=None)
