"""Does a super-resolution model other than FSRCNN help re-detect small keys?

Uses the same samples as probe_enlargement.py (real far keys, and clear frames
shrunk to make small keys). Each crop (padding 100% per side) is enlarged by
every method before YOLO:
- crop_only: no enlargement; YOLO's letterbox does it
- bicubic_xN: ordinary interpolation
- fsrcnn_xN: the current pipeline (x2 model repeated)
- <model>_xN: ESPCN / LapSRN / EDSR, downloaded on first use into models/

Reported per detector: keys found (IoU >= 0.3 at conf >= 0.25) and mean
matched confidence. Reported per method: median enlargement time on the CPU
(OpenCV dnn), which --threads can limit to imitate a low-spec device.

The key detector was trained on bicubic and FSRCNN enlargements, so the other
models are at a disadvantage with it; pass a second --weights trained without
them (or a COCO model with --label) to check that the ranking is not an
artifact of one detector.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe_enlargement import samples  # noqa: E402
from small_object_lab.metrics import Detection, intersection_over_union  # noqa: E402
from small_object_lab.super_resolution import (  # noqa: E402
    SR_MODEL_SPECS, FSRCNNSuperResolution, OpenCVSuperResolution, expanded_crop_region)


def annotated_samples(recording: Path, annotations: Path, factors: list[float], limit: int):
    """Samples from human-reviewed visible boxes instead of baseline detections.

    "real" keeps each frame as recorded; "synthetic_1/N" shrinks it (INTER_AREA,
    then JPEG quality 75 as in probe_enlargement) with the box scaled alike, so
    objects the baseline missed are included too.
    """
    rows = [r for r in csv.DictReader(annotations.open(encoding="utf-8-sig"))
            if r["reviewed"] == "1" and r["visible"] == "1"]
    if len(rows) > limit > 1:  # spread over the whole recording
        rows = [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]
    paths = {int(r["frame_id"]): r["path"] for r in csv.DictReader((recording / "frames.csv").open(encoding="utf-8"))}
    for factor in (1.0, *factors):
        for row in rows:
            image = cv2.imdecode(np.fromfile(recording / paths[int(row["frame_id"])], dtype=np.uint8), cv2.IMREAD_COLOR)
            x, y, width, height = (float(row[k]) for k in ("x", "y", "width", "height"))
            if factor == 1.0:
                yield "real", image, Detection("target", 1, round(x), round(y), max(1, round(width)), max(1, round(height)))
                continue
            small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)
            canvas = np.full_like(image, 128)
            canvas[:small.shape[0], :small.shape[1]] = small
            yield (f"synthetic_1/{round(1 / factor)}", canvas,
                   Detection("target", 1, round(x * factor), round(y * factor),
                             max(1, round(width * factor)), max(1, round(height * factor))))


def enlargers(names: list[str], scales: list[int]) -> dict:
    """variant name -> (function from crop to detector input, scale)."""
    out = {"crop_only": (lambda crop: crop, 1)}
    engines = {"fsrcnn": FSRCNNSuperResolution(ROOT / "models" / "FSRCNN_x2.pb")}
    engines.update({name: OpenCVSuperResolution(name, ROOT / "models") for name in names})
    for scale in scales:
        out[f"bicubic_x{scale}"] = (
            lambda crop, s=scale: cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), scale)
        for name, engine in engines.items():
            out[f"{name}_x{scale}"] = (lambda crop, e=engine, s=scale: e.enhance(crop, scale=s), scale)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("predictions", type=Path, nargs="?",
                        help="predictions.csv whose baseline rows give approximate truth")
    parser.add_argument("--annotations", type=Path,
                        help="Reviewed annotation CSV to use as truth instead of predictions.csv")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["espcn", "lapsrn"], choices=sorted(SR_MODEL_SPECS),
                        help="edsr is far heavier; add it only as an upper-bound reference")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4, 8])
    parser.add_argument("--label", help="Count only this class (needed for multi-class detectors such as COCO)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--padding", type=float, default=1.0)
    parser.add_argument("--min-time", type=float, default=24.0)
    parser.add_argument("--factors", type=float, nargs="*", default=[0.25, 1 / 6])
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--device", help="YOLO device (default: GPU if available)")
    parser.add_argument("--threads", type=int, help="CPU threads for the enlargement, e.g. 4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.threads:
        cv2.setNumThreads(args.threads)
    methods = enlargers(args.models, args.scales)
    for enlarge, _ in methods.values():  # download and initialise outside the timed region
        enlarge(np.zeros((32, 32, 3), np.uint8))
    detectors = {}
    for path in args.weights:
        name = path.parent.parent.name if path.parent.name == "weights" else path.stem
        detectors[name] = YOLO(str(path))
    found = collections.Counter()
    total = collections.Counter()
    confidence = collections.defaultdict(list)
    elapsed_ms = collections.defaultdict(list)
    crop_sides = collections.defaultdict(list)
    if (args.predictions is None) == (args.annotations is None):
        parser.error("give either predictions.csv or --annotations")
    if args.annotations:
        source = annotated_samples(args.recording, args.annotations, args.factors, args.limit)
    else:
        source = samples(args.recording, args.predictions, args.min_time, args.factors, args.limit)
    for group, frame, truth in source:
        h, w = frame.shape[:2]
        region = expanded_crop_region(truth, w, h, padding_ratio=args.padding)
        crop = frame[region.y:region.y + region.height, region.x:region.x + region.width]
        local_truth = Detection("target", 1, truth.x - region.x, truth.y - region.y, truth.width, truth.height)
        crop_sides[group].append(max(crop.shape[:2]))
        for variant, (enlarge, scale) in methods.items():
            started = time.perf_counter()
            image = enlarge(crop)
            elapsed_ms[(group, variant)].append(1000 * (time.perf_counter() - started))
            for detector_name, detector in detectors.items():
                result = detector.predict(image, conf=0.05, imgsz=args.imgsz, rect=False,
                                          device=args.device, verbose=False)[0]
                best = 0.0
                for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                                       result.boxes.conf.cpu().numpy(),
                                                       result.boxes.cls.cpu().numpy()):
                    if args.label and result.names[int(cls)] != args.label:
                        continue
                    box = Detection("target", float(conf), round(x1 / scale), round(y1 / scale),
                                    max(1, round((x2 - x1) / scale)), max(1, round((y2 - y1) / scale)))
                    if intersection_over_union(box, local_truth) >= 0.3:
                        best = max(best, float(conf))
                key = (detector_name, group, variant)
                total[key] += 1
                found[key] += best >= 0.25
                confidence[key].append(best)

    groups = sorted(crop_sides)
    if not groups:
        parser.exit(2, "Error: no samples (baseline rows with conf >= 0.6 in the expected time ranges)\n")
    report = {"crop_long_side_px_median": {g: float(np.median(crop_sides[g])) for g in groups},
              "enlargement_ms_median": {g: {v: round(float(np.median(elapsed_ms[(g, v)])), 2) for v in methods}
                                        for g in groups},
              "detectors": {}}
    for detector_name in detectors:
        print(f"\n== {detector_name}  (found@conf>=0.25 / n, mean matched conf, median enlargement ms)")
        print("variant".ljust(14) + "".join(g.rjust(31) for g in groups))
        for variant in methods:
            cells = []
            for group in groups:
                key = (detector_name, group, variant)
                mean = float(np.mean(confidence[key]))
                cells.append(f"{found[key]:3d}/{total[key]:<3d} {found[key] / total[key]:4.0%} c{mean:.2f}"
                             f" {report['enlargement_ms_median'][group][variant]:7.1f}ms".rjust(31))
                report["detectors"].setdefault(detector_name, {}).setdefault(group, {})[variant] = {
                    "found": found[key], "n": total[key], "mean_conf": round(mean, 4)}
            print(variant.ljust(14) + "".join(cells))
    print("\nmedian crop long side (px): " + ", ".join(f"{g}={report['crop_long_side_px_median'][g]:.0f}" for g in groups))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
