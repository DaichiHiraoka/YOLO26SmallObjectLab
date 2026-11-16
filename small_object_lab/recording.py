"""Lossless, bounded recording of decoded camera frames, before inference.

PNG preserves the pixels supplied by the camera decoder; it cannot undo the
camera's own JPEG compression. Timestamps describe PC receipt, not exposure.
This module deliberately does not import a detector or a model runtime.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MAX_BYTES = 10 * 1024**3
# While YOLO loads, the MJPEG reader stalls for ~1.4 s and then delivers ~25
# buffered frames at once; 16 slots dropped some of them (about 1.5 MB each).
DEFAULT_QUEUE_SIZE = 64


class RawRecorder:
    def __init__(
        self,
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if queue_size < 1 or max_bytes < 1:
            raise ValueError("queue_size and max_bytes must be positive")
        self.output_dir = Path(output_dir)
        self.session_id = session_id or self.output_dir.parent.name
        # Refuse even an empty existing directory: another run may own it.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "frames").mkdir()
        self._metadata = json.loads(json.dumps(metadata or {}, ensure_ascii=False))
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._queue: queue.Queue[tuple[int, float, np.ndarray]] = queue.Queue(queue_size)
        self._worker: threading.Thread | None = None
        self._closing = False
        self._closed = False
        self._accepting = True
        self._state = "waiting"
        self._sequence = 0
        self._saved_frames = 0
        self._dropped_frames = 0
        self._queue_drops = 0
        self._bytes_written = 0
        self._max_bytes = max_bytes
        self._errors: list[dict[str, object]] = []
        self._origin: float | None = None
        self._last_received: float | None = None
        self._dimensions: list[dict[str, int]] = []
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_manifest_write = 0.0
        self._complete = False
        with (self.output_dir / "frames.csv").open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            csv.writer(handle).writerow(
                ["frame_id", "timestamp_s", "path", "width", "height"]
            )
        self._write_manifest()

    def submit(self, frame: np.ndarray, timestamp_s: float | None = None) -> int:
        """Queue a private copy, returning its 1-based ID even when dropped.

        ``timestamp_s`` is a monotonically increasing receipt-clock value. The
        first value becomes zero in frames.csv. A full queue never blocks the
        camera or inference; its dropped frame IDs remain gaps in the CSV.
        """
        received = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(received):
            raise ValueError("timestamp_s must be finite")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 numpy image")
        if frame.ndim not in (2, 3) or not all(frame.shape[:2]):
            raise ValueError("frame must have nonempty image dimensions")
        if frame.ndim == 3 and frame.shape[2] not in (3, 4):
            raise ValueError("frame must be grayscale, BGR or BGRA")
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("recorder is closed")
            if self._last_received is not None and received < self._last_received:
                raise ValueError("timestamp_s must not move backwards")
            self._last_received = received
            if self._origin is None:
                self._origin = received
            self._sequence += 1
            frame_id = self._sequence
            if not self._accepting:
                self._dropped_frames += 1
                return frame_id
            if self._queue.full():
                self._dropped_frames += 1
                self._queue_drops += 1
                return frame_id
            try:
                self._queue.put_nowait((frame_id, received - self._origin, frame.copy()))
            except queue.Full:
                self._dropped_frames += 1
                self._queue_drops += 1
            if self._worker is None:
                self._state = "recording"
                self._worker = threading.Thread(
                    target=self._write_frames, name="raw-frame-recorder", daemon=True
                )
                self._worker.start()
            return frame_id

    def record_settings(self, settings: dict[str, object]) -> None:
        """Retain live setting changes without writing on the capture thread."""
        snapshot = json.loads(json.dumps(settings, ensure_ascii=False))
        with self._lock:
            if self._closing or self._closed:
                return
            relative_time = (
                max(0.0, time.monotonic() - self._origin)
                if self._origin is not None else 0.0
            )
            self._metadata.setdefault("settings_history", []).append(
                {"timestamp_s": relative_time, "settings": snapshot}
            )

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "state": self._state,
                "recording": self._accepting and not self._closing,
                "closing": self._closing and not self._closed,
                "complete": self._complete,
                "duration_s": (
                    self._last_received - self._origin
                    if self._origin is not None and self._last_received is not None
                    else 0.0
                ),
                "submitted_frames": self._sequence,
                "saved_frames": self._saved_frames,
                "dropped_frames": self._dropped_frames,
                "queue_drops": self._queue_drops,
                "pending_frames": self._queue.qsize(),
                "bytes_written": self._bytes_written,
                "max_bytes": self._max_bytes,
                "errors": list(self._errors),
                "output_dir": str(self.output_dir.resolve()),
            }

    def close(self) -> None:
        """Finish queued writes and persist the final manifest (idempotent)."""
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closing = True
                self._accepting = False
                worker = self._worker
            if worker is not None:
                worker.join()
            with self._lock:
                self._closed = True
                self._complete = not self._errors
                if self._state not in {"limit_reached", "error"}:
                    self._state = "closed"
            self._write_manifest()

    def _manifest(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1,
                "session_id": self.session_id,
                "format": "png_sequence",
                "complete": self._complete,
                "state": self._state,
                "started_at_utc": self._started_at,
                "timestamp_basis": "monotonic PC receipt seconds relative to first frame",
                "dropped_frames": self._dropped_frames,
                "queue_drops": self._queue_drops,
                "submitted_frames": self._sequence,
                "saved_frames": self._saved_frames,
                "bytes_written": self._bytes_written,
                "max_bytes": self._max_bytes,
                "errors": list(self._errors),
                "source_dimensions": list(self._dimensions),
                "metadata": json.loads(json.dumps(self._metadata, ensure_ascii=False)),
            }

    def _write_manifest(self) -> None:
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(self._manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output_dir / "manifest.json")
        self._last_manifest_write = time.monotonic()

    def _record_error(self, frame_id: int | None, error: Exception) -> None:
        with self._lock:
            self._state = "error"
            self._accepting = False
            self._errors.append({"frame_id": frame_id, "message": str(error)})

    def _write_frames(self) -> None:
        try:
            with (self.output_dir / "frames.csv").open(
                "a", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                while True:
                    try:
                        frame_id, timestamp, frame = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        with self._lock:
                            if self._closing or not self._accepting:
                                break
                        continue
                    saved = False
                    try:
                        with self._lock:
                            stopped = self._state in {"limit_reached", "error"}
                        if stopped:
                            with self._lock:
                                self._dropped_frames += 1
                            continue
                        ok, encoded = cv2.imencode(
                            ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                        )
                        if not ok:
                            raise OSError("PNG encoding failed")
                        payload = encoded.tobytes()
                        with self._lock:
                            if self._bytes_written + len(payload) > self._max_bytes:
                                self._state = "limit_reached"
                                self._accepting = False
                                self._dropped_frames += 1
                                continue
                        relative_path = f"frames/{frame_id:08d}.png"
                        with (self.output_dir / relative_path).open("xb") as image_file:
                            image_file.write(payload)
                        height, width = frame.shape[:2]
                        writer.writerow(
                            [frame_id, f"{timestamp:.9f}", relative_path, width, height]
                        )
                        stream.flush()
                        with self._lock:
                            self._saved_frames += 1
                            self._bytes_written += len(payload)
                            dimensions = {"width": width, "height": height}
                            if dimensions not in self._dimensions:
                                self._dimensions.append(dimensions)
                        saved = True
                        if time.monotonic() - self._last_manifest_write >= 1.0:
                            self._write_manifest()
                    except Exception as error:
                        with self._lock:
                            if not saved:
                                self._dropped_frames += 1
                        self._record_error(frame_id, error)
                    finally:
                        self._queue.task_done()
        except Exception as error:
            self._record_error(None, error)
        finally:
            # In particular, opening frames.csv may fail before entering the
            # loop. Account for every accepted-but-unwritten queued frame.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                with self._lock:
                    self._dropped_frames += 1
                self._queue.task_done()
            try:
                self._write_manifest()
            except Exception as error:
                self._record_error(None, error)
