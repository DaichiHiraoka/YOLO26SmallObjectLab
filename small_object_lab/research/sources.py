"""Read immutable local recordings or videos without dropping inference frames."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class InputFrame:
    frame_id: int
    timestamp_s: float
    image: np.ndarray


class FrameSource:
    def __init__(self, source: Path, *, allow_incomplete: bool = False) -> None:
        self.path = Path(source).resolve(strict=True)
        if self.path.is_file() and self.path.name == "manifest.json":
            self.path = self.path.parent
        if self.path.is_dir() and not (self.path / "manifest.json").exists():
            legacy = self.path / "recording"
            if not legacy.is_dir():
                found = sorted(p.parent.name for p in self.path.glob("recordings/*/manifest.json"))
                raise ValueError("Specify one recording directory: "
                                 f"{self.path / 'recordings' / '<recording_id>'} "
                                 f"(found: {', '.join(found) or 'none'})")
            self.path = legacy.resolve(strict=True)
        self.rows: list[dict] | None = None
        self.metadata: dict = {}
        if self.path.is_dir():
            manifest_path = self.path / "manifest.json"
            self.metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self.metadata.get("schema_version") != 1 or self.metadata.get("format") != "png_sequence":
                raise ValueError("Unsupported recording format")
            if not allow_incomplete and (not self.metadata.get("complete") or
                    self.metadata.get("dropped_frames", 0) or self.metadata.get("errors")):
                raise ValueError("Recording is incomplete or has dropped frames/errors; use --allow-incomplete only for diagnostic runs")
            self.session_id = str(self.metadata["session_id"])
            with (self.path / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.rows = list(csv.DictReader(stream))
            last_id, last_time = 0, -1.0
            for row in self.rows:
                row["frame_id"] = int(row["frame_id"])
                row["timestamp_s"] = float(row["timestamp_s"])
                row["width"], row["height"] = int(row["width"]), int(row["height"])
                image_path = (self.path / row["path"]).resolve(strict=True)
                if not image_path.is_relative_to(self.path):
                    raise ValueError("Recording image path escapes its directory")
                if (row["frame_id"] <= last_id or not math.isfinite(row["timestamp_s"])
                        or row["timestamp_s"] < 0 or row["timestamp_s"] < last_time
                        or min(row["width"], row["height"]) <= 0):
                    raise ValueError("Invalid recording frame index, timestamp, or dimensions")
                if not allow_incomplete and row["frame_id"] != last_id + 1:
                    raise ValueError("Recording has missing frame indices")
                row["image_path"] = image_path
                last_id, last_time = row["frame_id"], row["timestamp_s"]
            if not self.rows:
                raise ValueError("Recording contains no frames")
            digest = hashlib.sha256(manifest_path.read_bytes() + (self.path / "frames.csv").read_bytes())
            for row in self.rows:
                digest.update(bytes.fromhex(sha256_file(row["image_path"])))
            self.fingerprint = digest.hexdigest()
        else:
            if self.path.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mov"}:
                raise ValueError("Source must be a recording directory or a local video")
            self.fingerprint = sha256_file(self.path)
            self.session_id = f"{self.path.stem}-{self.fingerprint[:12]}"
            self.metadata = {"format": "video", "timing": "nominal_frame_rate"}

    def __iter__(self) -> Iterator[InputFrame]:
        if self.rows is not None:
            for row in self.rows:
                # imdecode handles Unicode paths on Windows consistently.
                image = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (row["height"], row["width"]):
                    raise ValueError(f"Invalid image at frame {row['frame_id']}")
                yield InputFrame(row["frame_id"], row["timestamp_s"], image)
            return
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open local video")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Video has no valid frame rate")
            index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                index += 1
                yield InputFrame(index, (index - 1) / fps, image)
            if not index:
                raise ValueError("Video contains no decodable frames")
        finally:
            capture.release()


ANNOTATION_FIELDS = ["session_id", "frame_id", "object_id", "label", "visible",
                     "x", "y", "width", "height", "reviewed"]


def create_annotation_template(source: FrameSource, output: Path, *, label: str = "cup") -> int:
    """An empty review queue, never machine guesses presented as ground truth."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        count = 0
        for frame in source:
            writer.writerow(dict(session_id=source.session_id, frame_id=frame.frame_id,
                                 object_id="1", label=label, reviewed=0))
            count += 1
    return count
