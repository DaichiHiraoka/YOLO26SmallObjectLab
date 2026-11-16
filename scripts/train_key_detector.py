"""Fine-tune YOLO26 into a single-class key detector with a light GPU load.

The COCO-pretrained backbone is frozen, the batch is small and AMP is on, so
only the neck/head is trained. Results go to a new folder; nothing is overwritten.

Example:
    .venv\\Scripts\\python.exe scripts\\train_key_detector.py --data datasets/key_o365_leftincar_bydate/data.yaml --name key_o365_leftincar_bydate_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--name", required=True, help="Run name under --project (must be new)")
    parser.add_argument("--project", type=Path, default=Path("outputs/key_detector"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10, help="Number of leading (backbone) layers kept frozen")
    parser.add_argument("--workers", type=int, default=2)
    # Small keys vanish when mosaic (4 images in one) and +/-50% scaling shrink them further.
    parser.add_argument("--scale", type=float, default=0.5, help="Random scale gain range (Ultralytics default 0.5)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability (Ultralytics default 1.0)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Final epochs without mosaic")
    # Keys must stay detectable when crop/SR enlarges them, not only at their training size.
    parser.add_argument("--multi-scale", type=float, default=0.0,
                        help="Vary the training input size by +/- this fraction of --imgsz")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.project / args.name
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    model_path = args.model.resolve(strict=True)  # Local weights only; never download a substitute.
    data_path = args.data.resolve(strict=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch=args.batch, freeze=args.freeze, workers=args.workers, device=args.device,
        scale=args.scale, mosaic=args.mosaic, close_mosaic=args.close_mosaic, multi_scale=args.multi_scale,
        amp=True, cache=False, seed=args.seed, deterministic=True,
        project=str(args.project.resolve()), name=args.name, exist_ok=False, plots=True,
    )
    manifest = data_path.parent / "manifest.json"
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "data": {"yaml": str(data_path), "manifest_sha256": sha256_file(manifest) if manifest.exists() else None},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "load_profile": "small batch, AMP, cache off" + (", frozen backbone" if args.freeze else ""),
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best weights: {(run_dir / 'weights' / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
