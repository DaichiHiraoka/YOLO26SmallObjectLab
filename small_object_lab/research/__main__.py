"""Run with python -m small_object_lab.research --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..super_resolution import DEFAULT_SR_MODEL, SR_MODEL_NAMES
from .runner import ALL_METHODS, PADDINGS
from .sources import FrameSource, create_annotation_template


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Local recording, paired evaluation and crop/SR selector research")
    sub = command.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="Run all comparison methods on identical local frames")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path, required=True, help="New output directory (never overwritten)")
    replay.add_argument("--methods", nargs="+", default=["baseline", "fixed", "bicubic", "rule"],
                        choices=list(ALL_METHODS),
                        help="'<method>+rd' adds appearance re-discovery of a lost target")
    root = Path(__file__).resolve().parents[2]
    replay.add_argument("--model", type=Path, default=root / "yolo26n.pt")
    replay.add_argument("--sr-model", default=DEFAULT_SR_MODEL, choices=list(SR_MODEL_NAMES),
                        help="Super-resolution engine for 'sr' actions (fsrcnn reproduces runs before Nov 2026)")
    replay.add_argument("--device", default="auto")
    replay.add_argument("--label", default="cup")
    replay.add_argument("--confidence", type=float, default=0.20)
    replay.add_argument("--imgsz", type=int, default=960)
    replay.add_argument("--local-imgsz", type=int, default=640)
    replay.add_argument("--fixed-scale", type=int, choices=[2, 4, 8], default=4)
    replay.add_argument("--fixed-padding", type=float, choices=list(PADDINGS), default=0.2)
    replay.add_argument("--retention-s", type=float, default=0.75)
    replay.add_argument("--rediscover-threshold", type=float, default=0.75,
                        help="Appearance similarity needed to re-adopt a lost target (+rd methods)")
    replay.add_argument("--max-frames", type=int)
    replay.add_argument("--collect-candidates", action="store_true", help="Measure all feasible crop/scale candidates for later supervised training")
    replay.add_argument("--selector", type=Path)
    replay.add_argument("--allow-training-session", action="store_true", help="Diagnostic only: permit replay on training session")
    replay.add_argument("--allow-incomplete", action="store_true", help="Diagnostic only: allow incomplete/dropped input recordings")
    for name, help_text in [("annotation-template", "Create UNREVIEWED ground-truth rows"),
                            ("annotate", "Review local recording boxes in an OpenCV window")]:
        annotation = sub.add_parser(name, help=help_text)
        annotation.add_argument("source", type=Path)
        annotation.add_argument("--output", required=True, type=Path)
        annotation.add_argument("--label", default="cup")
        annotation.add_argument("--allow-incomplete", action="store_true")
        if name == "annotate":
            annotation.add_argument("--start-frame", type=int, help="First frame_id to review")
            annotation.add_argument("--end-frame", type=int, help="Last frame_id to review")
            annotation.add_argument("--stride", type=int, default=1, help="Review every Nth frame in the range")
            annotation.add_argument("--proposals", type=Path,
                                    help="Draft boxes from 'propose': accept/reject them in a grid first")
    propose = sub.add_parser("propose", help="Draft boxes with larger detectors for human review (not ground truth)")
    propose.add_argument("source", type=Path)
    propose.add_argument("--output", required=True, type=Path)
    propose.add_argument("--label", default="cup")
    propose.add_argument("--models", nargs="+", default=["yolo26x.pt", "yolo26l.pt"],
                         help="Draft detectors; use models other than the one being compared")
    propose.add_argument("--imgsz", type=int, default=1280)
    propose.add_argument("--device", default="auto")
    propose.add_argument("--allow-incomplete", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Score predictions against reviewed ground truth")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("annotations", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--iou", type=float, default=0.5)
    train = sub.add_parser("train-selector", help="Train a lightweight joint crop/scale model using reviewed training data")
    train.add_argument("samples", type=Path)
    train.add_argument("annotations", type=Path)
    train.add_argument("split", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--latency-penalty", type=float, default=0.002)
    train.add_argument("--ridge-alpha", type=float, default=1.0,
                       help="Shrinkage of each action's utility model; choose it on validation sessions")
    train.add_argument("--frames", choices=("decision", "all"), default="decision",
                       help="decision: only frames where the enhancement trigger fires (as at run time)")
    assemble = sub.add_parser("assemble", help="Combine candidate runs and reviewed annotation files")
    assemble.add_argument("--runs", nargs="+", type=Path, required=True)
    assemble.add_argument("--annotations", nargs="+", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command == "replay":
            from .runner import ExperimentConfig, InferenceBackend, run_experiment
            from .selector import LearnedSelector
            source = FrameSource(args.source, allow_incomplete=args.allow_incomplete)
            config = ExperimentConfig(label=args.label, confidence=args.confidence, imgsz=args.imgsz,
                                      local_imgsz=args.local_imgsz, retention_s=args.retention_s,
                                      fixed_scale=args.fixed_scale, fixed_padding=args.fixed_padding,
                                      sr_model=args.sr_model)
            from .rediscovery import RediscoveryConfig
            if any(m.startswith("learned") for m in args.methods) and args.selector is None:
                command.error("--methods learned requires --selector")
            selector = LearnedSelector.load(args.selector) if args.selector else None
            backend = InferenceBackend(args.model, args.sr_model, args.device)
            result = run_experiment(source, args.output, backend, config, methods=tuple(args.methods),
                                    selector=selector, max_frames=args.max_frames,
                                    collect_candidates=args.collect_candidates,
                                    allow_training_session=args.allow_training_session,
                                    rediscovery=RediscoveryConfig(threshold=args.rediscover_threshold))
            print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                              "processed_frames": result["processed_frames"], "summary": result["summary"]},
                             ensure_ascii=False, indent=2))
        elif args.command in {"annotation-template", "annotate"}:
            source = FrameSource(args.source, allow_incomplete=args.allow_incomplete)
            if args.command == "annotate":
                from .annotation import review_annotations
                review_annotations(source, args.output, label=args.label, start_frame=args.start_frame,
                                   end_frame=args.end_frame, stride=args.stride, proposals_path=args.proposals)
            else:
                count = create_annotation_template(source, args.output, label=args.label)
                print(json.dumps({"session_id": source.session_id, "unreviewed_frames": count,
                                  "output": str(args.output.resolve())}, ensure_ascii=False))
        elif args.command == "propose":
            from .proposals import combined_predictor, write_proposals, yolo_predictor
            source = FrameSource(args.source, allow_incomplete=args.allow_incomplete)
            root = Path(__file__).resolve().parents[2]
            models = [str(root / m) if (root / m).exists() else m for m in args.models]
            predict = combined_predictor([yolo_predictor(m, args.label, args.device) for m in models])
            print(json.dumps(write_proposals(source, args.output, predict, model_name="+".join(args.models),
                                             label=args.label, imgsz=args.imgsz), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            from .evaluation import evaluate_predictions
            result = evaluate_predictions(args.predictions, args.annotations, iou_threshold=args.iou)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            print(f"Evaluation written: {args.output.resolve()}")
        elif args.command == "train-selector":
            from .selector import train_selector
            result = train_selector(args.samples, args.annotations, args.split, args.output,
                                    latency_penalty=args.latency_penalty, frames=args.frames,
                                    ridge_alpha=args.ridge_alpha)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "assemble":
            from .dataset import assemble_dataset
            print(json.dumps(assemble_dataset(args.runs, args.annotations, args.output), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        command.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
