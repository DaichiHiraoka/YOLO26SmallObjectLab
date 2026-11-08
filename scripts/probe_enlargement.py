"""Why does enlarging a crop before YOLO lower detection? Isolate the causes.

For each small key, the same crop (padding 100% per side) is turned into the
detector input in several ways:
- yolo_internal: pass the crop; YOLO's letterbox does the only enlargement
- 1step_<kernel>: resize once so the long side is exactly --imgsz (YOLO then
  does not rescale), isolating the interpolation kernel
- 2step_<method>: enlarge first (bicubic or FSRCNN xN), then let YOLO rescale
  (the current pipeline)
- fsrcnn_xN_then_linear: FSRCNN, then one exact linear resize to --imgsz

Ground truth is the baseline full-frame detection (real video, conf >= 0.6) or
that box scaled down with the frame (synthetic small keys). A key is found if
a prediction overlaps it with IoU >= 0.3 at conf >= 0.25.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from small_object_lab.metrics import Detection, intersection_over_union  # noqa: E402
from small_object_lab.super_resolution import FSRCNNSuperResolution, expanded_crop_region  # noqa: E402

KERNELS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
           "lanczos": cv2.INTER_LANCZOS4}


def to_long_side(image: np.ndarray, side: int, interpolation: int) -> np.ndarray:
    scale = side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=interpolation)


def variants(crop: np.ndarray, sr: FSRCNNSuperResolution, imgsz: int) -> dict[str, tuple[np.ndarray, float]]:
    """name -> (detector input, factor from detector input back to crop pixels)."""
    out = {"yolo_internal": (crop, 1.0)}
    for name, kernel in KERNELS.items():
        image = to_long_side(crop, imgsz, kernel)
        out[f"1step_{name}"] = (image, crop.shape[1] / image.shape[1])
    for scale in (2, 4, 8):
        out[f"2step_bicubic_x{scale}"] = (cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), 1 / scale)
        enhanced = sr.enhance(crop, scale=scale)
        out[f"2step_fsrcnn_x{scale}"] = (enhanced, 1 / scale)
        exact = to_long_side(enhanced, imgsz, cv2.INTER_LINEAR if max(enhanced.shape[:2]) < imgsz else cv2.INTER_AREA)
        out[f"fsrcnn_x{scale}_then_linear"] = (exact, crop.shape[1] / exact.shape[1])
    return out


def samples(recording: Path, predictions: Path, min_time: float, factors: list[float], limit: int):
    rows = [r for r in csv.DictReader(predictions.open(encoding="utf-8-sig"))
            if r["method"] == "baseline" and r["observed"] == "1" and float(r["confidence"]) >= 0.6]
    real = [r for r in rows if float(r["timestamp_s"]) >= min_time][:limit]
    for r in real:
        image = cv2.imread(str(recording / f"frames/{int(r['frame_id']):08d}.png"))
        yield "real_far", image, Detection("key", 1, int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"]))
    clear = [r for r in rows if 9 <= float(r["timestamp_s"]) < 30][:limit]
    for factor in factors:
        for r in clear:
            image = cv2.imread(str(recording / f"frames/{int(r['frame_id']):08d}.png"))
            small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)
            canvas = np.full_like(image, 128)
            canvas[:small.shape[0], :small.shape[1]] = small
            truth = Detection("key", 1, round(int(r["x"]) * factor), round(int(r["y"]) * factor),
                              max(1, round(int(r["width"]) * factor)), max(1, round(int(r["height"]) * factor)))
            yield f"synthetic_1/{round(1 / factor)}", canvas, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("predictions", type=Path, help="predictions.csv whose baseline rows give approximate truth")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--padding", type=float, default=1.0)
    parser.add_argument("--min-time", type=float, default=24.0)
    parser.add_argument("--factors", type=float, nargs="*", default=[0.25, 1 / 6])
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from ultralytics import YOLO

    sr = FSRCNNSuperResolution()
    models = {path.parent.parent.name: YOLO(str(path)) for path in args.weights}
    found = collections.Counter()
    total = collections.Counter()
    confidence = collections.defaultdict(list)
    for group, frame, truth in samples(args.recording, args.predictions, args.min_time, args.factors, args.limit):
        h, w = frame.shape[:2]
        region = expanded_crop_region(truth, w, h, padding_ratio=args.padding)
        crop = frame[region.y:region.y + region.height, region.x:region.x + region.width]
        local_truth = Detection("key", 1, truth.x - region.x, truth.y - region.y, truth.width, truth.height)
        for variant, (image, back) in variants(crop, sr, args.imgsz).items():
            for model_name, model in models.items():
                result = model.predict(image, conf=0.05, imgsz=args.imgsz, rect=False, device=0, verbose=False)[0]
                best = 0.0
                for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    box = Detection("key", float(conf), round(x1 * back), round(y1 * back),
                                    max(1, round((x2 - x1) * back)), max(1, round((y2 - y1) * back)))
                    if intersection_over_union(box, local_truth) >= 0.3:
                        best = max(best, float(conf))
                key = (model_name, group, variant)
                total[key] += 1
                found[key] += best >= 0.25
                confidence[key].append(best)
    report = {}
    groups = sorted({k[1] for k in total})
    for model_name in models:
        print(f"\n== {model_name}  (found@conf>=0.25 / n, mean matched conf)")
        names = list(variants(np.zeros((40, 60, 3), np.uint8), sr, args.imgsz))
        print("variant".ljust(28) + "".join(g.rjust(22) for g in groups))
        for variant in names:
            cells = []
            for group in groups:
                key = (model_name, group, variant)
                rate = found[key] / total[key] if total[key] else 0
                mean = float(np.mean(confidence[key])) if confidence[key] else 0
                cells.append(f"{found[key]:3d}/{total[key]:<3d} {rate:4.0%} c{mean:.2f}".rjust(22))
                report.setdefault(model_name, {}).setdefault(group, {})[variant] = {
                    "found": found[key], "n": total[key], "mean_conf": round(mean, 4)}
            print(variant.ljust(28) + "".join(cells))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
