"""Draft ground-truth boxes for human review (never ground truth by themselves).

Drafts come from a larger detector than the one being compared, so the
reviewer is not simply shown the evaluated pipeline's own output. Per frame:
1. the large model on the full frame;
2. if that misses, the same model on a plain crop (no super resolution)
   around the last drafted position, so small far objects are still drafted;
3. short gaps between two nearly identical drafts are interpolated.
Every draft keeps its source so the review and the thesis can report how
each confirmed box was obtained.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .sources import FrameSource

PROPOSAL_FIELDS = ["session_id", "frame_id", "x", "y", "width", "height", "confidence", "source"]
Box = tuple[int, int, int, int]
# (image, imgsz) -> [(x, y, width, height, confidence)] for the target label only
Predict = Callable[[np.ndarray, int], list[tuple[float, float, float, float, float]]]


@dataclass
class Proposal:
    frame_id: int
    box: Box
    confidence: float
    source: str  # "full", "crop" or "interpolated"


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _clip(box: Iterable[float], width: int, height: int) -> Box | None:
    x, y, w, h = box
    x1, y1 = max(0, round(x)), max(0, round(y))
    x2, y2 = min(width, round(x + w)), min(height, round(y + h))
    return (x1, y1, x2 - x1, y2 - y1) if x2 > x1 and y2 > y1 else None


def _closest(found: list[tuple[Box, float]], anchor: Box | None) -> tuple[Box, float]:
    if anchor is None:
        return max(found, key=lambda item: item[1])
    ax, ay = anchor[0] + anchor[2] / 2, anchor[1] + anchor[3] / 2
    return min(found, key=lambda item: (item[0][0] + item[0][2] / 2 - ax) ** 2
               + (item[0][1] + item[0][3] / 2 - ay) ** 2 - 1e-6 * item[1])


def _near(box: Box, anchor: Box) -> bool:
    """Single-object assumption: reject a draft that jumps far from the last one."""
    reach = 3 * max(anchor[2], anchor[3], 20)
    return (abs(box[0] + box[2] / 2 - anchor[0] - anchor[2] / 2) <= reach
            and abs(box[1] + box[3] / 2 - anchor[1] - anchor[3] / 2) <= reach)


def propose_frames(frames: Iterable[tuple[int, np.ndarray]], predict: Predict, *, imgsz: int = 1280,
                   crop_imgsz: int = 640, memory_frames: int = 30, max_gap: int = 15,
                   min_gap_iou: float = 0.3) -> dict[int, Proposal]:
    proposals: dict[int, Proposal] = {}
    anchor: Proposal | None = None
    order: list[int] = []
    for frame_id, image in frames:
        order.append(frame_id)
        height, width = image.shape[:2]
        if anchor is not None and frame_id - anchor.frame_id > memory_frames:
            anchor = None
        found = [(box, conf) for *raw, conf in predict(image, imgsz) if (box := _clip(raw, width, height))]
        if anchor is not None:
            found = [item for item in found if _near(item[0], anchor.box)]
        source = "full"
        if not found and anchor is not None:
            ax, ay, aw, ah = anchor.box
            side = min(max(160, 4 * max(aw, ah)), width, height)
            cx, cy = ax + aw / 2, ay + ah / 2
            x0 = int(min(max(0, cx - side / 2), width - side))
            y0 = int(min(max(0, cy - side / 2), height - side))
            crop = image[y0:y0 + side, x0:x0 + side]
            found = [(box, conf) for *raw, conf in predict(crop, crop_imgsz)
                     if (box := _clip((raw[0] + x0, raw[1] + y0, raw[2], raw[3]), width, height))]
            found = [item for item in found if _near(item[0], anchor.box)]
            source = "crop"
        if found:
            box, conf = _closest(found, anchor.box if anchor else None)
            anchor = proposals[frame_id] = Proposal(frame_id, box, float(conf), source)
    # Interpolate short gaps between two drafts of an object that barely moved.
    drafted = [f for f in order if f in proposals]
    for left, right in zip(drafted, drafted[1:]):
        a, b = proposals[left], proposals[right]
        missing = [f for f in order if left < f < right]
        if not missing or len(missing) > max_gap or _iou(a.box, b.box) < min_gap_iou:
            continue
        for f in missing:
            t = (f - left) / (right - left)
            box = tuple(round(p + (q - p) * t) for p, q in zip(a.box, b.box))
            proposals[f] = Proposal(f, box, min(a.confidence, b.confidence), "interpolated")
    return proposals


def yolo_predictor(model_path: Path | str, label: str, device: str = "auto") -> Predict:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    classes = [index for index, name in names.items() if name == label]
    if not classes:
        raise ValueError(f"Model {model_path} has no class named {label!r}")
    target = (0 if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def predict(image: np.ndarray, imgsz: int):
        result = model.predict(image, imgsz=imgsz, conf=0.25, classes=classes, device=target, verbose=False)[0]
        return [(x1, y1, x2 - x1, y2 - y1, conf) for (x1, y1, x2, y2), conf
                in zip(result.boxes.xyxy.cpu().numpy().tolist(), result.boxes.conf.cpu().numpy().tolist())]
    return predict


def combined_predictor(predictors: list[Predict]) -> Predict:
    """Union of several models' boxes; the single-object choice picks one later."""
    def predict(image: np.ndarray, imgsz: int):
        return [box for predictor in predictors for box in predictor(image, imgsz)]
    return predict


def write_proposals(source: FrameSource, output: Path, predict: Predict, *, model_name: str, label: str,
                    imgsz: int = 1280) -> dict:
    if source.rows is None:
        raise ValueError("Proposals need a PNG recording directory")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new name")

    def frames():
        for row in source.rows:
            image = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read frame {row['frame_id']}")
            yield row["frame_id"], image

    proposals = propose_frames(frames(), predict, imgsz=imgsz)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROPOSAL_FIELDS)
        writer.writeheader()
        for frame_id in sorted(proposals):
            p = proposals[frame_id]
            writer.writerow(dict(session_id=source.session_id, frame_id=frame_id, x=p.box[0], y=p.box[1],
                                 width=p.box[2], height=p.box[3], confidence=round(p.confidence, 4),
                                 source=p.source))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    counts = {kind: sum(p.source == kind for p in proposals.values()) for kind in ("full", "crop", "interpolated")}
    summary = {"schema_version": 1, "session_id": source.session_id, "source_sha256": source.fingerprint,
               "model": model_name, "label": label, "imgsz": imgsz, "frames": len(source.rows),
               "drafted": len(proposals), "by_source": counts,
               "note": "Drafts only. A box becomes ground truth only after a person accepts or redraws it."}
    Path(str(output) + ".meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_proposals(path: Path, session_id: str) -> dict[int, Proposal]:
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["session_id"] != session_id:
                raise ValueError("Proposal file belongs to a different recording")
            frame_id = int(row["frame_id"])
            result[frame_id] = Proposal(frame_id, tuple(int(row[k]) for k in ("x", "y", "width", "height")),
                                        float(row["confidence"]), row["source"])
    return result
