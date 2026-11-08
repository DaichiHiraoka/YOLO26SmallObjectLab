"""Recall of a key detector by box size on a YOLO-format split (default: Objects365+LeftInCAR val).

Size is sqrt(box area) in original-image pixels. A ground-truth key counts as
found when an unmatched prediction overlaps it with IoU >= --iou.

Example:
    .venv\\Scripts\\python.exe scripts\\eval_key_detector.py outputs/key_detector/key_o365_leftincar_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BINS = ((0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, float("inf"), ">=64px"))


def iou(a, b) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/key_o365_leftincar_bydate"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve(strict=True)))
    counts = {name: [0, 0] for _, _, name in BINS}
    false_positives = 0
    for image_path in sorted(p for p in (args.dataset / "images" / args.split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        truths = []
        label = args.dataset / "labels" / args.split / f"{image_path.stem}.txt"
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                truths.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                               (cx + bw / 2) * width, (cy + bh / 2) * height))
        result = model.predict(image, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        predictions = [tuple(map(float, box)) for box in result.boxes.xyxy.cpu().numpy()]
        used: set[int] = set()
        for truth in truths:
            side = ((truth[2] - truth[0]) * (truth[3] - truth[1])) ** 0.5
            name = next(name for low, high, name in BINS if low <= side < high)
            counts[name][1] += 1
            score, index = max(((iou(truth, p), i) for i, p in enumerate(predictions) if i not in used),
                               default=(0.0, None))
            if score >= args.iou:
                counts[name][0] += 1
                used.add(index)
        false_positives += len(predictions) - len(used)
    found = sum(c[0] for c in counts.values())
    total = sum(c[1] for c in counts.values())
    print(json.dumps({
        "weights": str(args.weights), "split": args.split, "conf": args.conf, "iou": args.iou,
        "recall": f"{found}/{total}", "by_size": {k: f"{d}/{n}" for k, (d, n) in counts.items()},
        "false_positives": false_positives,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
