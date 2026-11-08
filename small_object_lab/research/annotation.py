"""Manual, resumable review of one object's boxes in a lossless recording."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .evaluation import load_annotations
from .sources import ANNOTATION_FIELDS, FrameSource


class AnnotationReview:
    def __init__(self, source: FrameSource, output: Path, label: str = "cup") -> None:
        if source.rows is None:
            raise ValueError("The review window currently accepts PNG recording directories, not encoded videos")
        self.source, self.output, self.label = source, Path(output), label
        self.rows = {row["frame_id"]: dict(session_id=source.session_id, frame_id=row["frame_id"],
                    object_id="1", label=label, visible="", x="", y="", width="", height="", reviewed=0)
                     for row in source.rows}
        if self.output.exists():
            existing = load_annotations(self.output)
            for (session, frame_id), row in existing.items():
                if session != source.session_id or frame_id not in self.rows or row["label"] != label:
                    raise ValueError("Annotation file does not belong to this recording/label")
                row = dict(row)
                row["reviewed"], row["visible"] = int(row["reviewed"]), int(row["visible"])
                self.rows[frame_id] = {key: "" if value is None else value for key, value in row.items()}
        self.dimensions = {row["frame_id"]: (row["width"], row["height"]) for row in source.rows}
        metadata_path = Path(str(self.output) + ".meta.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("source_sha256") != source.fingerprint:
                raise ValueError("Recording contents changed since this annotation review was created")

    def commit(self, frame_id: int, *, box: tuple[int, int, int, int] | None = None,
               visible: bool = True, reviewed: bool = True) -> None:
        row = dict(self.rows[frame_id])
        if visible and reviewed:
            if box is None:
                raise ValueError("Draw a box before confirming a visible object")
            x, y, width, height = box
            image_width, image_height = self.dimensions[frame_id]
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > image_width or y + height > image_height:
                raise ValueError("Box must stay inside the original frame")
            row.update(x=int(x), y=int(y), width=int(width), height=int(height))
        else:
            row.update(x="", y="", width="", height="")
        row.update(visible=int(visible) if reviewed else "", reviewed=int(reviewed))
        self.rows[frame_id] = row
        self.save()

    def save(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(self.output) + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows.values())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.output)
        metadata_path = Path(str(self.output) + ".meta.json")
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps({"schema_version": 1, "source_sha256": self.source.fingerprint,
                    "session_id": self.source.session_id, "coordinate_space": "original_pixels"}, indent=2), encoding="utf-8")


def review_annotations(source: FrameSource, output: Path, *, label: str = "cup",
                       start_frame: int | None = None, end_frame: int | None = None, stride: int = 1) -> None:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    review = AnnotationReview(source, output, label)
    # Only the selected frames are shown; all others stay unreviewed and unscored.
    frames = [row for row in source.rows
              if (start_frame is None or row["frame_id"] >= start_frame)
              and (end_frame is None or row["frame_id"] <= end_frame)][::stride]
    if not frames:
        raise ValueError("No frames in the selected range")
    index = next((i for i, item in enumerate(frames) if not review.rows[item["frame_id"]]["reviewed"]), 0)
    previous_box = None
    window = "Ground truth review - Q quits"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while index < len(frames):
            frame = frames[index]
            image = cv2.imdecode(np.fromfile(frame["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read frame {frame['frame_id']}")
            scale = min(1.0, 1100 / image.shape[1], 750 / image.shape[0])
            preview = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            row = review.rows[frame["frame_id"]]
            selected = tuple(round(float(row[k])) for k in ("x", "y", "width", "height")) if row["width"] != "" else None
            message = ""
            while True:
                display = preview.copy()
                if selected:
                    x, y, w, h = (round(v*scale) for v in selected)
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,220,255), 2)
                banner = np.zeros((96, display.shape[1], 3), dtype=np.uint8)
                lines = [f"Frame {frame['frame_id']} ({index+1}/{len(frames)})   Reviewed: {row['reviewed']}",
                         "R draw | C copy previous | S confirm + next | I invisible + next",
                         "N skip | P previous | U mark unreviewed | Q quit", message]
                for i, text in enumerate(lines):
                    cv2.putText(banner, text, (8, 20 + i*23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
                cv2.imshow(window, np.vstack((banner, display)))
                key = cv2.waitKey(50) & 0xFF
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (ord('q'), 27):
                    return
                if key == ord('r'):
                    roi = cv2.selectROI("Draw the object's visible extent", preview, showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow("Draw the object's visible extent")
                    if roi[2] > 0 and roi[3] > 0:
                        x, y, w, h = (round(v/scale) for v in roi)
                        selected = (x, y, min(w, frame["width"]-x), min(h, frame["height"]-y))
                elif key == ord('c'):
                    selected = previous_box
                    message = "Copied draft: verify it, then press S to confirm."
                elif key in (ord('s'), ord('i'), ord('u')):
                    try:
                        review.commit(frame["frame_id"], box=selected, visible=key != ord('i'), reviewed=key != ord('u'))
                        if key == ord('s'):
                            previous_box = selected
                        index += 1
                        break
                    except ValueError as error:
                        message = str(error)
                elif key == ord('n'):
                    index += 1
                    break
                elif key == ord('p'):
                    index = max(0, index-1)
                    break
    finally:
        # Only explicit confirmations set reviewed=1. Merely browsing never does.
        review.save()
        cv2.destroyAllWindows()
