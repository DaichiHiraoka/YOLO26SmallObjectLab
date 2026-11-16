"""Resumable human review of one object's boxes in a lossless recording.

With draft proposals (see proposals.py) the reviewer first accepts or rejects
drafts in a grid, then confirms in a second grid which remaining frames do not
show the object at all, and draws boxes one by one only where it is visible.
Nothing becomes ground truth without an explicit human decision, and the
source of every confirmed box is kept in <output>.review.json.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .evaluation import load_annotations
from .proposals import Proposal, load_proposals
from .sources import ANNOTATION_FIELDS, FrameSource

GRID_COLUMNS, GRID_ROWS, TILE = 4, 3, 240
ABSENCE_COLUMNS, ABSENCE_ROWS, ABSENCE_TILE = 4, 2, 320


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
        # frame_id -> how the confirmed decision was made; rejected drafts await manual review
        self.provenance: dict[int, str] = {}
        self.rejected: set[int] = set()
        # frames the reviewer said do show the object, so they need a hand-drawn box
        self.flagged: set[int] = set()
        if self.review_path.exists():
            data = json.loads(self.review_path.read_text(encoding="utf-8"))
            self.provenance = {int(k): v for k, v in data.get("provenance", {}).items()}
            self.rejected = set(data.get("rejected_drafts", []))
            self.flagged = set(data.get("flagged_visible", []))

    @property
    def review_path(self) -> Path:
        return Path(str(self.output) + ".review.json")

    def commit(self, frame_id: int, *, box: tuple[int, int, int, int] | None = None,
               visible: bool = True, reviewed: bool = True, provenance: str | None = None) -> None:
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
        if reviewed:
            self.provenance[frame_id] = provenance or ("manual_box" if visible else "invisible")
            self.flagged.discard(frame_id)
        else:
            self.provenance.pop(frame_id, None)
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
        counts: dict[str, int] = {}
        for value in self.provenance.values():
            counts[value] = counts.get(value, 0) + 1
        self.review_path.write_text(json.dumps({
            "schema_version": 1, "summary": counts, "rejected_drafts": sorted(self.rejected),
            "flagged_visible": sorted(self.flagged),
            "provenance": {str(k): v for k, v in sorted(self.provenance.items())}}, indent=1), encoding="utf-8")


def tile_image(image: np.ndarray, proposal: Proposal, *, rejected: bool, reviewed: bool) -> np.ndarray:
    """Zoomed view of one draft with enough surroundings to judge the box."""
    height, width = image.shape[:2]
    x, y, w, h = proposal.box
    side = min(max(120, 3 * max(w, h)), width, height)
    x0 = int(min(max(0, x + w / 2 - side / 2), width - side))
    y0 = int(min(max(0, y + h / 2 - side / 2), height - side))
    gain = TILE / side
    tile = cv2.resize(image[y0:y0 + side, x0:x0 + side], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
    color = (0, 220, 255) if proposal.source == "interpolated" else (0, 255, 0)
    cv2.rectangle(tile, (round((x - x0) * gain), round((y - y0) * gain)),
                  (round((x + w - x0) * gain), round((y + h - y0) * gain)), color, 1)
    text = f"{proposal.frame_id} {proposal.source[:6]}"
    cv2.rectangle(tile, (0, 0), (8 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if rejected:
        cv2.line(tile, (0, 0), (TILE - 1, TILE - 1), (0, 0, 255), 3)
        cv2.line(tile, (TILE - 1, 0), (0, TILE - 1), (0, 0, 255), 3)
    elif reviewed:
        cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (0, 160, 0), 3)
    return tile


def apply_grid_page(review: AnnotationReview, page: list[int], proposals: dict[int, Proposal],
                    rejected: set[int]) -> None:
    """Accepted drafts become confirmed boxes; rejected ones go to manual review."""
    for frame_id in page:
        if frame_id in rejected:
            review.rejected.add(frame_id)
            review.commit(frame_id, reviewed=False)
        else:
            review.rejected.discard(frame_id)
            review.commit(frame_id, box=proposals[frame_id].box,
                          provenance=f"accepted_draft_{proposals[frame_id].source}")


def grid_queue(review: AnnotationReview, frames: list[dict], proposals: dict[int, Proposal]) -> list[int]:
    return [f["frame_id"] for f in frames if f["frame_id"] in proposals
            and not review.rows[f["frame_id"]]["reviewed"] and f["frame_id"] not in review.rejected]


def _grid_review(review: AnnotationReview, frames: list[dict], proposals: dict[int, Proposal]) -> bool:
    """Returns False if the reviewer quit."""
    ids = grid_queue(review, frames, proposals)
    if not ids:
        return True
    paths = {f["frame_id"]: f["image_path"] for f in frames}
    per_page = GRID_COLUMNS * GRID_ROWS
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    window = "Draft review - click wrong boxes, Space accepts the rest"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
    cv2.setMouseCallback(window, on_mouse)
    banner_height, index = 70, 0
    try:
        while index < len(pages):
            page = pages[index]
            images = {f: cv2.imdecode(np.fromfile(paths[f], dtype=np.uint8), cv2.IMREAD_COLOR) for f in page}
            rejected = {f for f in page if f in review.rejected}
            while True:
                tiles = [tile_image(images[f], proposals[f], rejected=f in rejected,
                                    reviewed=bool(review.rows[f]["reviewed"])) for f in page]
                tiles += [np.zeros((TILE, TILE, 3), np.uint8)] * (per_page - len(tiles))
                grid = np.vstack([np.hstack(tiles[r * GRID_COLUMNS:(r + 1) * GRID_COLUMNS]) for r in range(GRID_ROWS)])
                banner = np.zeros((banner_height, grid.shape[1], 3), np.uint8)
                for i, text in enumerate([f"Page {index + 1}/{len(pages)}",
                                          "Click a tile whose box is wrong or loose (red X = manual review later)",
                                          "Space/Enter accept page | P previous page | Q quit (progress is saved)"]):
                    cv2.putText(banner, text, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window, np.vstack((banner, grid)))
                key = cv2.waitKey(50) & 0xFF
                for x, y in clicked:
                    cell = ((y - banner_height) // TILE) * GRID_COLUMNS + x // TILE
                    if y >= banner_height and cell < len(page):
                        rejected ^= {page[cell]}
                clicked.clear()
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (ord('q'), 27):
                    return False
                if key in (ord(' '), 13, ord('s')):
                    apply_grid_page(review, page, proposals, rejected)
                    index += 1
                    break
                if key == ord('p') and index > 0:
                    index -= 1
                    break
    finally:
        cv2.destroyWindow(window)
    return True


def absence_queue(review: AnnotationReview, frames: list[dict], proposals: dict[int, Proposal]) -> list[int]:
    """Unreviewed frames without an accepted draft that the reviewer has not yet flagged."""
    return [f["frame_id"] for f in frames if not review.rows[f["frame_id"]]["reviewed"]
            and f["frame_id"] not in review.flagged
            and (f["frame_id"] not in proposals or f["frame_id"] in review.rejected)]


def apply_absence_page(review: AnnotationReview, page: list[int], has_object: set[int]) -> None:
    """Unclicked frames are confirmed as not showing the object; clicked ones await a drawn box."""
    for frame_id in page:
        if frame_id in has_object:
            # also undoes an earlier "no object" for this frame (reviewer went back with P)
            if review.rows[frame_id]["reviewed"]:
                review.commit(frame_id, reviewed=False)
            review.flagged.add(frame_id)
        else:
            review.commit(frame_id, visible=False, provenance="grid_not_visible")
    review.save()


def _absence_grid(review: AnnotationReview, frames: list[dict], proposals: dict[int, Proposal], label: str) -> bool:
    """Whole frames, several per page: the reviewer clicks those where the object is visible.

    Returns False if the reviewer quit."""
    ids = absence_queue(review, frames, proposals)
    if not ids:
        return True
    paths = {f["frame_id"]: f["image_path"] for f in frames}
    per_page = ABSENCE_COLUMNS * ABSENCE_ROWS
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    window = f"No-{label} check - click frames that DO show the {label}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
    cv2.setMouseCallback(window, on_mouse)
    banner_height, index, size = 70, 0, ABSENCE_TILE
    try:
        while index < len(pages):
            page = pages[index]
            images = {}
            for f in page:
                image = cv2.imdecode(np.fromfile(paths[f], dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = image.shape[:2]
                gain = size / max(h, w)
                fitted = cv2.resize(image, (round(w * gain), round(h * gain)), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((size, size, 3), np.uint8)
                canvas[:fitted.shape[0], :fitted.shape[1]] = fitted
                images[f] = canvas
            has_object = {f for f in page if f in review.flagged}
            while True:
                tiles = []
                for f in page:
                    tile = images[f].copy()
                    text = f"{f}" + (f"  {label.upper()} VISIBLE" if f in has_object else "")
                    cv2.rectangle(tile, (0, 0), (8 + 9 * len(text), 22), (0, 0, 0), -1)
                    cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    if f in has_object:
                        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (0, 140, 255), 5)
                    tiles.append(tile)
                tiles += [np.zeros((size, size, 3), np.uint8)] * (per_page - len(tiles))
                grid = np.vstack([np.hstack(tiles[r * ABSENCE_COLUMNS:(r + 1) * ABSENCE_COLUMNS])
                                  for r in range(ABSENCE_ROWS)])
                banner = np.zeros((banner_height, grid.shape[1], 3), np.uint8)
                for i, text in enumerate([f"Page {index + 1}/{len(pages)}   Frames without an accepted draft",
                                          f"Click every frame where the {label} is visible, even partly (orange = draw later)",
                                          f"Space/Enter: the other frames have NO {label} | P previous page | Q quit"]):
                    cv2.putText(banner, text, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(window, np.vstack((banner, grid)))
                key = cv2.waitKey(50) & 0xFF
                for x, y in clicked:
                    cell = ((y - banner_height) // size) * ABSENCE_COLUMNS + x // size
                    if y >= banner_height and cell < len(page):
                        has_object ^= {page[cell]}
                clicked.clear()
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (ord('q'), 27):
                    return False
                if key in (ord(' '), 13, ord('s')):
                    apply_absence_page(review, page, has_object)
                    index += 1
                    break
                if key == ord('p') and index > 0:
                    index -= 1
                    break
    finally:
        cv2.destroyWindow(window)
    return True


def review_annotations(source: FrameSource, output: Path, *, label: str = "cup",
                       start_frame: int | None = None, end_frame: int | None = None, stride: int = 1,
                       proposals_path: Path | None = None) -> None:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    review = AnnotationReview(source, output, label)
    # Only the selected frames are shown; all others stay unreviewed and unscored.
    frames = [row for row in source.rows
              if (start_frame is None or row["frame_id"] >= start_frame)
              and (end_frame is None or row["frame_id"] <= end_frame)][::stride]
    if not frames:
        raise ValueError("No frames in the selected range")
    proposals = load_proposals(proposals_path, source.session_id) if proposals_path else {}
    if proposals:
        if not _grid_review(review, frames, proposals) or not _absence_grid(review, frames, proposals, label):
            review.save()
            return
        # One-by-one review of rejected drafts and frames without a draft, at the same stride,
        # so the reviewed sample is not biased toward frames the draft detector found easy.
        frames = [f for f in frames if not review.rows[f["frame_id"]]["reviewed"]]
        if not frames:
            print("All selected frames are reviewed.")
            return
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
            draft = proposals.get(frame["frame_id"])
            message = ""
            if frame["frame_id"] in review.flagged:
                message = f"You marked this frame as showing the {label}: draw it with R, then S."
            elif selected is None and draft is not None:
                selected = draft.box
                message = f"Draft ({draft.source}) was rejected: redraw with R, or I if not visible."
            elif draft is None and proposals:
                message = "No draft for this frame: draw with R, or I if not visible."
            while True:
                display = preview.copy()
                if selected:
                    x, y, w, h = (round(v*scale) for v in selected)
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,220,255), 2)
                banner = np.zeros((96, display.shape[1], 3), dtype=np.uint8)
                lines = [f"Frame {frame['frame_id']} ({index+1}/{len(frames)})   Reviewed: {row['reviewed']}",
                         "R draw | C copy previous | S confirm + next | I invisible + next",
                         "N skip (NOT saved) | P previous | U mark unreviewed | Q quit", message]
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
                        same = draft is not None and selected == draft.box
                        review.commit(frame["frame_id"], box=selected, visible=key != ord('i'), reviewed=key != ord('u'),
                                      provenance=None if key == ord('i') else
                                      f"single_view_draft_{draft.source}" if same else "manual_box")
                        review.rejected.discard(frame["frame_id"])
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
