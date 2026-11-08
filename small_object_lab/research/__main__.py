"""Run with python -m small_object_lab.research --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import PADDINGS
from .sources import FrameSource, create_annotation_template


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Local recording, paired evaluation and crop/SR selector research")
    sub = command.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="Run all comparison methods on identical local frames")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path, required=True, help="New output directory (never overwritten)")
    replay.add_argument("--methods", nargs="+", default=["baseline", "fixed", "bicubic", "rule"],
                        choices=["baseline", "fixed", "bicubic", "rule", "learned"])
    root = Path(__file__).resolve().parents[2]
    replay.add_argument("--model", type=Path, default=root / "yolo26n.pt")
    replay.add_argument("--sr-model", type=Path, default=root / "models" / "FSRCNN_x2.pb")
    replay.add_argument("--device", default="auto")
    replay.add_argument("--label", default="cup")
    replay.add_argument("--confidence", type=float, default=0.20)
    replay.add_argument("--imgsz", type=int, default=960)
    replay.add_argument("--local-imgsz", type=int, default=640)
    replay.add_argument("--fixed-scale", type=int, choices=[2, 4, 8], default=4)
    replay.add_argument("--fixed-padding", type=float, choices=list(PADDINGS), default=0.2)
    replay.add_argument("--retention-s", type=float, default=0.75)
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
                                      fixed_scale=args.fixed_scale, fixed_padding=args.fixed_padding)
            if "learned" in args.methods and args.selector is None:
                command.error("--methods learned requires --selector")
            selector = LearnedSelector.load(args.selector) if args.selector else None
            backend = InferenceBackend(args.model, args.sr_model, args.device)
            result = run_experiment(source, args.output, backend, config, methods=tuple(args.methods),
                                    selector=selector, max_frames=args.max_frames,
                                    collect_candidates=args.collect_candidates,
                                    allow_training_session=args.allow_training_session)
            print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                              "processed_frames": result["processed_frames"], "summary": result["summary"]},
                             ensure_ascii=False, indent=2))
        elif args.command in {"annotation-template", "annotate"}:
            source = FrameSource(args.source, allow_incomplete=args.allow_incomplete)
            if args.command == "annotate":
                from .annotation import review_annotations
                review_annotations(source, args.output, label=args.label, start_frame=args.start_frame,
                                   end_frame=args.end_frame, stride=args.stride)
            else:
                count = create_annotation_template(source, args.output, label=args.label)
                print(json.dumps({"session_id": source.session_id, "unreviewed_frames": count,
                                  "output": str(args.output.resolve())}, ensure_ascii=False))
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
                                    latency_penalty=args.latency_penalty)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "assemble":
            from .dataset import assemble_dataset
            print(json.dumps(assemble_dataset(args.runs, args.annotations, args.output), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        command.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
