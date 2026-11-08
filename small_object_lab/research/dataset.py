"""Assemble candidate logs and reviewed annotations without silently mixing runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluation import ANNOTATION_FIELDS, load_annotations
from .sources import sha256_file


def assemble_dataset(runs: list[Path], annotations: list[Path], output: Path) -> dict:
    if not runs or not annotations:
        raise ValueError("Candidate runs and annotation files are required")
    seen = set()
    sessions = set()
    sample_paths = []
    for run in runs:
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or not manifest.get("collect_candidates"):
            raise ValueError(f"Not a completed candidate collection: {run}")
        if manifest.get("training_session_override"):
            raise ValueError("Diagnostic training-session replays cannot be a dataset input")
        path = run / "candidates.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["session_id"], row["frame_id"])
                if row["session_id"] != manifest["session_id"] or key in seen:
                    raise ValueError("Duplicate or inconsistent sample session/frame")
                seen.add(key)
                sessions.add(key[0])
        sample_paths.append(path)
    gt = {}
    for path in annotations:
        for key, row in load_annotations(path).items():
            if key in gt:
                raise ValueError(f"Duplicate annotation: {key}")
            if key[0] not in sessions:
                raise ValueError(f"Annotation session has no candidate run: {key[0]}")
            gt[key] = row
    if not seen:
        raise ValueError("No candidate frames")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as destination:
        for path in sample_paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    destination.write(line.rstrip("\r\n") + "\n")
    with (output / "annotations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for key in sorted(gt):
            row = dict(gt[key])
            row["reviewed"], row["visible"] = int(row["reviewed"]), int(row["visible"])
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    # A list to assign deliberately by filming session, not a fabricated random frame split.
    (output / "sessions.json").write_text(json.dumps(sorted(sessions), ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"schema_version": 1, "sample_frames": len(seen), "annotation_frames": len(gt),
              "reviewed_frames": sum(r["reviewed"] for r in gt.values()), "sessions": sorted(sessions),
              "inputs": {str(p.resolve()): sha256_file(p) for p in [*sample_paths, *annotations]}}
    (output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
