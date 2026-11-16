"""Exercise local weights on historical snapshots; NOT a research evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from small_object_lab.metrics import Detection
from small_object_lab.recording import RawRecorder
from small_object_lab.research.runner import Action, ExperimentConfig, InferenceBackend, execute_action, run_experiment
from small_object_lab.research.sources import FrameSource, create_annotation_template
from small_object_lab.research.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sr-model", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted(args.images.glob("auto_*.jpg"))
    if not files:
        raise ValueError("No historical snapshots found")
    recorder = RawRecorder(args.output / "smoke-input" / "recording", metadata={
        "purpose": "software_smoke_only_not_research_data",
        "warning": "Historical snapshots already contain overlays; not valid detection/tracking ground truth",
        "source_images": [str(p.resolve()) for p in files],
    })
    try:
        for index, path in enumerate(files):
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable snapshot: {path}")
            recorder.submit(image, timestamp_s=float(index))
    finally:
        recorder.close()
    source = FrameSource(recorder.output_dir)
    config = ExperimentConfig()
    backend = InferenceBackend(args.model, args.sr_model)
    result = run_experiment(source, args.output / "comparison", backend, config, collect_candidates=True)
    # Exercise each enhancement factor even when the detector chooses no recovery action.
    image = cv2.imdecode(np.fromfile(files[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
    # Recorded cup box in auto_0004 of the July 23 session. It is only a
    # hand-selected smoke-test ROI, never a reviewed annotation.
    box = Detection("cup", 0.5, 263, 294, 29, 46)
    forced_actions = [execute_action(backend, image, box, None, Action("sr", 0.2, scale), config)
                      for scale in (2, 4, 8)]
    annotations = args.output / "UNREVIEWED.csv"
    create_annotation_template(source, annotations)
    evaluation = evaluate_predictions(args.output / "comparison" / "predictions.csv", annotations)
    assert all(group["detection"]["recall"] is None for group in evaluation["groups"])
    report = {"purpose": "software_smoke_only_not_research_result", "device": backend.device,
              "gpu_name": backend.torch.cuda.get_device_name() if backend.torch.cuda.is_available() else None,
              "frames": result["processed_frames"], "comparison_status": result["status"],
              "summary": result["summary"], "forced_sr_actions": forced_actions,
              "unreviewed_accuracy_is_null": True, "recording": recorder.status()}
    (args.output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("purpose", "device", "frames", "comparison_status", "unreviewed_accuracy_is_null")}, indent=2))


if __name__ == "__main__":
    main()
