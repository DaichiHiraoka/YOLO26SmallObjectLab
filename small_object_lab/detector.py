from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import Detection, StabilityTracker, intersection_over_union, smallest
from .super_resolution import (
    CropRegion,
    FSRCNNSuperResolution,
    LargestObjectInstabilityMonitor,
    SuperResolutionDecision,
    expanded_crop_region,
    remap_from_super_resolution,
)


@dataclass
class LabSettings:
    confidence: float = 0.20
    imgsz: int = 960
    target_classes: str = ""
    stable_frames: int = 3
    super_resolution_enabled: bool = True
    sr_trigger_confidence: float = 0.45
    sr_trigger_area_pct: float = 1.0

    def class_filter(self) -> set[str]:
        return {
            token.strip().casefold()
            for token in self.target_classes.split(",")
            if token.strip()
        }


class FrameMailbox:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> int:
        with self._condition:
            self._sequence += 1
            self._frame = frame
            self._condition.notify()
            return self._sequence

    def wait_for_next(
        self, previous_sequence: int, timeout: float = 1.0
    ) -> tuple[int, np.ndarray | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > previous_sequence, timeout=timeout
            )
            return self._sequence, self._frame


class SmallObjectLab:
    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        device: str = "auto",
        source: str = "",
        output_root: Path = Path("outputs/yolo26_small_object_lab"),
        auto_save_interval_sec: float = 15.0,
    ) -> None:
        if auto_save_interval_sec <= 0:
            raise ValueError("auto_save_interval_sec must be greater than 0")

        self.model_name = model_name
        self.device = device
        self.source = source
        self.output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "detections.csv"
        self.measurements_path = self.output_dir / "measurements.csv"
        self.auto_save_interval_sec = auto_save_interval_sec

        self.mailbox = FrameMailbox()
        self.lock = threading.RLock()
        self.measurements_lock = threading.Lock()
        self.settings = LabSettings()
        self.stability = StabilityTracker()
        self.super_resolution = FSRCNNSuperResolution()
        self.instability_monitor = LargestObjectInstabilityMonitor()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.source_worker: threading.Thread | None = None

        self.model: Any = None
        self.status_message = "YOLO26を読み込み中"
        self.error_message = ""
        self.camera_connected = False
        self.uploaded_frames = 0
        self.processed_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.inference_ms = 0.0
        self.inference_fps = 0.0
        self.last_frame_at = 0.0
        self.latest_jpeg: bytes | None = None
        self.latest_detections: list[Detection] = []
        self.smallest_current: Detection | None = None
        self.smallest_confirmed: Detection | None = None
        self.next_auto_save_at: float | None = None
        self.auto_save_count = 0
        self.last_auto_save_path = ""
        self.super_resolution_active = False
        self.super_resolution_reason = ""
        self.super_resolution_attempts = 0
        self.super_resolution_adoptions = 0
        self.super_resolution_ms = 0.0
        self.super_resolution_region: CropRegion | None = None
        self.super_resolution_target: Detection | None = None
        self.super_resolution_error = ""

        self._write_csv_headers()

    def _write_csv_headers(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.writer(stream).writerow(
                [
                    "timestamp",
                    "frame",
                    "label",
                    "detection_source",
                    "confidence",
                    "x",
                    "y",
                    "width_px",
                    "height_px",
                    "area_px",
                    "width_pct",
                    "height_pct",
                    "area_pct",
                    "streak",
                    "imgsz",
                    "inference_ms",
                    "source_width",
                    "source_height",
                ]
            )
        with self.measurements_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            csv.writer(stream).writerow(
                [
                    "timestamp",
                    "mode",
                    "detection_status",
                    "frame",
                    "object_name",
                    "physical_width_mm",
                    "camera_distance_cm",
                    "label",
                    "detection_source",
                    "confidence",
                    "x",
                    "y",
                    "width_px",
                    "height_px",
                    "area_px",
                    "area_pct",
                    "streak",
                    "target_classes",
                    "imgsz",
                    "source_width",
                    "source_height",
                    "image_file",
                ]
            )

    def start(self) -> None:
        self.worker = threading.Thread(
            target=self._run, name="yolo26-inference", daemon=True
        )
        self.worker.start()
        if self.source:
            self.source_worker = threading.Thread(
                target=self._read_source,
                name="smartphone-camera-source",
                daemon=True,
            )
            self.source_worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=3.0)
        if self.source_worker is not None:
            self.source_worker.join(timeout=3.0)

    def submit_jpeg(self, payload: bytes) -> bool:
        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.submit_frame(frame)
        return True

    def submit_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.uploaded_frames += 1
            self.camera_connected = True
            self.last_frame_at = time.monotonic()
        self.mailbox.put(frame)

    def update_settings(self, values: dict[str, object]) -> LabSettings:
        with self.lock:
            confidence = float(values.get("confidence", self.settings.confidence))
            imgsz = int(values.get("imgsz", self.settings.imgsz))
            stable_frames = int(
                values.get("stable_frames", self.settings.stable_frames)
            )
            target_classes = str(
                values.get("target_classes", self.settings.target_classes)
            )
            super_resolution_enabled = bool(
                values.get(
                    "super_resolution_enabled",
                    self.settings.super_resolution_enabled,
                )
            )
            sr_trigger_confidence = float(
                values.get(
                    "sr_trigger_confidence",
                    self.settings.sr_trigger_confidence,
                )
            )
            sr_trigger_area_pct = float(
                values.get(
                    "sr_trigger_area_pct",
                    self.settings.sr_trigger_area_pct,
                )
            )
            if not 0.01 <= confidence <= 0.99:
                raise ValueError("confidenceは0.01〜0.99で指定してください")
            if imgsz not in {320, 480, 640, 960, 1280, 1536}:
                raise ValueError("imgszは320, 480, 640, 960, 1280, 1536から選択してください")
            if not 1 <= stable_frames <= 30:
                raise ValueError("stable_framesは1〜30で指定してください")
            if not 0.01 <= sr_trigger_confidence <= 0.99:
                raise ValueError("超解像confidenceは0.01〜0.99で指定してください")
            if not 0.01 <= sr_trigger_area_pct <= 100.0:
                raise ValueError("超解像bbox面積は0.01〜100%で指定してください")
            self.settings = LabSettings(
                confidence=confidence,
                imgsz=imgsz,
                target_classes=target_classes,
                stable_frames=stable_frames,
                super_resolution_enabled=super_resolution_enabled,
                sr_trigger_confidence=sr_trigger_confidence,
                sr_trigger_area_pct=sr_trigger_area_pct,
            )
            return self.settings

    def reset_record(self) -> None:
        with self.lock:
            self.smallest_confirmed = None
            self.smallest_current = None
            self.stability.reset()
            self.instability_monitor.reset()

    def record_measurement(self, values: dict[str, object]) -> Path:
        with self.lock:
            detection = self.smallest_current or self.smallest_confirmed
            if detection is None:
                raise ValueError("記録できる検出結果がありません")
            data = detection.as_dict(self.frame_width, self.frame_height)
            row = [
                datetime.now().isoformat(timespec="milliseconds"),
                "manual",
                (
                    "stable"
                    if detection.streak >= self.settings.stable_frames
                    else "unstable"
                ),
                self.processed_frames,
                str(values.get("object_name", "")).strip(),
                str(values.get("physical_width_mm", "")).strip(),
                str(values.get("camera_distance_cm", "")).strip(),
                detection.label,
                "super_resolution" if detection.enhanced else "full_frame",
                detection.confidence,
                detection.x,
                detection.y,
                detection.width,
                detection.height,
                detection.area_px,
                data["area_pct"],
                detection.streak,
                self.settings.target_classes,
                self.settings.imgsz,
                self.frame_width,
                self.frame_height,
                "",
            ]
        with self.measurements_lock:
            with self.measurements_path.open(
                "a", newline="", encoding="utf-8-sig"
            ) as stream:
                csv.writer(stream).writerow(row)
        return self.measurements_path

    def snapshot(self) -> Path:
        with self.lock:
            if self.latest_jpeg is None:
                raise ValueError("保存できる映像がありません")
            payload = self.latest_jpeg
        path = self.output_dir / f"snapshot_{datetime.now():%H%M%S_%f}.jpg"
        path.write_bytes(payload)
        return path

    def status(self) -> dict[str, object]:
        with self.lock:
            connected = (
                self.camera_connected
                and time.monotonic() - self.last_frame_at < 3.0
            )
            return {
                "model": self.model_name,
                "device": self._resolved_device(),
                "source_mode": "android-camera-streamer",
                "source": self.source,
                "model_ready": self.model is not None,
                "status": self.status_message,
                "error": self.error_message,
                "camera_connected": connected,
                "uploaded_frames": self.uploaded_frames,
                "processed_frames": self.processed_frames,
                "source_width": self.frame_width,
                "source_height": self.frame_height,
                "inference_ms": round(self.inference_ms, 1),
                "inference_fps": round(self.inference_fps, 2),
                "auto_save_interval_sec": self.auto_save_interval_sec,
                "auto_save_count": self.auto_save_count,
                "auto_save_in_sec": (
                    round(
                        max(0.0, self.next_auto_save_at - time.monotonic()),
                        1,
                    )
                    if self.next_auto_save_at is not None
                    else self.auto_save_interval_sec
                ),
                "last_auto_save_path": self.last_auto_save_path,
                "super_resolution": {
                    "enabled": self.settings.super_resolution_enabled,
                    "model": "FSRCNN x2",
                    "target_strategy": "largest_bbox",
                    "ready": self.super_resolution.ready,
                    "active": self.super_resolution_active,
                    "reason": self.super_resolution_reason,
                    "attempts": self.super_resolution_attempts,
                    "adoptions": self.super_resolution_adoptions,
                    "inference_ms": round(self.super_resolution_ms, 1),
                    "error": self.super_resolution_error,
                    "target": (
                        self.super_resolution_target.as_dict(
                            self.frame_width, self.frame_height
                        )
                        if self.super_resolution_target
                        else None
                    ),
                },
                "settings": vars(self.settings),
                "detections": [
                    item.as_dict(self.frame_width, self.frame_height)
                    for item in self.latest_detections
                ],
                "smallest_current": (
                    self.smallest_current.as_dict(
                        self.frame_width, self.frame_height
                    )
                    if self.smallest_current
                    else None
                ),
                "smallest_confirmed": (
                    self.smallest_confirmed.as_dict(
                        self.frame_width, self.frame_height
                    )
                    if self.smallest_confirmed
                    else None
                ),
                "output_dir": str(self.output_dir.resolve()),
            }

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _read_source(self) -> None:
        while not self.stop_event.is_set():
            capture = cv2.VideoCapture(self.source)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                with self.lock:
                    self.status_message = "android-camera-streamerへ再接続中"
                    self.error_message = f"MJPEGを開けません: {self.source}"
                capture.release()
                self.stop_event.wait(1.0)
                continue

            with self.lock:
                self.status_message = "android-camera-streamerへ接続"
                self.error_message = ""
            while not self.stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    break
                self.submit_frame(frame)
            capture.release()
            if not self.stop_event.is_set():
                with self.lock:
                    self.camera_connected = False
                    self.status_message = "android-camera-streamerへ再接続中"
                self.stop_event.wait(1.0)

    def _run(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
            with self.lock:
                self.status_message = "スマホカメラ待機中"
        except Exception as error:
            with self.lock:
                self.status_message = "モデル読込エラー"
                self.error_message = str(error)
            return

        sequence = 0
        while not self.stop_event.is_set():
            next_sequence, frame = self.mailbox.wait_for_next(sequence)
            if next_sequence == sequence or frame is None:
                continue
            sequence = next_sequence
            try:
                self._infer(frame)
            except Exception as error:
                with self.lock:
                    self.status_message = "推論エラー"
                    self.error_message = str(error)

    def _infer(self, frame: np.ndarray) -> None:
        started = time.perf_counter()
        with self.lock:
            settings = self.settings
        results = self.model.predict(
            frame,
            conf=settings.confidence,
            imgsz=settings.imgsz,
            device=self._resolved_device(),
            verbose=False,
        )
        height, width = frame.shape[:2]
        detections = self._extract_detections(results, settings)
        decision = self.instability_monitor.decide(
            detections,
            frame_width=width,
            frame_height=height,
            trigger_confidence=settings.sr_trigger_confidence,
            trigger_area_pct=settings.sr_trigger_area_pct,
        )
        sr_region: CropRegion | None = None
        sr_attempted = False
        sr_used = False
        sr_elapsed = 0.0
        sr_error = ""
        sr_target = decision.target
        if settings.super_resolution_enabled and decision.should_apply:
            sr_attempted = True
            sr_started = time.perf_counter()
            try:
                enhanced_detection, sr_region = self._infer_super_resolution(
                    frame,
                    decision,
                    settings,
                )
                detections, sr_used = self._merge_super_resolution_detection(
                    detections,
                    decision.target,
                    enhanced_detection,
                )
                if sr_used:
                    sr_target = enhanced_detection
            except Exception as error:
                sr_error = str(error)
            sr_elapsed = time.perf_counter() - sr_started

        detections = self.stability.update(detections)
        elapsed = time.perf_counter() - started
        current_smallest = smallest(detections)
        confirmed = [
            item for item in detections if item.streak >= settings.stable_frames
        ]

        annotated = frame.copy()
        if sr_region is not None:
            cv2.rectangle(
                annotated,
                (sr_region.x, sr_region.y),
                (
                    sr_region.x + sr_region.width,
                    sr_region.y + sr_region.height,
                ),
                (255, 130, 20),
                2,
            )
            cv2.putText(
                annotated,
                f"FSRCNN x2: {decision.reason}",
                (sr_region.x, max(20, sr_region.y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 130, 20),
                2,
                cv2.LINE_AA,
            )
        for detection in detections:
            is_confirmed = detection.streak >= settings.stable_frames
            if detection.enhanced:
                color = (255, 80, 220)
            else:
                color = (54, 211, 153) if is_confirmed else (0, 190, 255)
            cv2.rectangle(
                annotated,
                (detection.x, detection.y),
                (
                    detection.x + detection.width,
                    detection.y + detection.height,
                ),
                color,
                2,
            )
            label = (
                f"{detection.label} {detection.confidence:.2f} "
                f"{detection.width}x{detection.height}px "
                f"[{detection.streak}/{settings.stable_frames}]"
                f"{' SR' if detection.enhanced else ''}"
            )
            cv2.putText(
                annotated,
                label,
                (detection.x, max(20, detection.y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        ok, jpeg = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            return

        with self.lock:
            self.processed_frames += 1
            frame_number = self.processed_frames
            self.frame_width = width
            self.frame_height = height
            self.inference_ms = elapsed * 1000.0
            self.inference_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.latest_detections = detections
            self.smallest_current = current_smallest
            for item in confirmed:
                if (
                    self.smallest_confirmed is None
                    or item.area_px < self.smallest_confirmed.area_px
                ):
                    self.smallest_confirmed = item
            self.latest_jpeg = jpeg.tobytes()
            self.super_resolution_active = sr_attempted
            self.super_resolution_reason = decision.reason if sr_attempted else ""
            self.super_resolution_ms = sr_elapsed * 1000.0
            self.super_resolution_region = sr_region
            self.super_resolution_target = sr_target if sr_attempted else None
            self.super_resolution_error = sr_error
            if sr_attempted:
                self.super_resolution_attempts += 1
            if sr_used:
                self.super_resolution_adoptions += 1
            self.status_message = "局所超解像中" if sr_attempted else "推論中"
            self.error_message = ""

        self._append_detections(
            detections,
            frame_number,
            width,
            height,
            settings,
            elapsed * 1000.0,
        )
        self._maybe_auto_save(
            frame_number=frame_number,
            detections=detections,
            confirmed=confirmed,
            settings=settings,
            jpeg_payload=jpeg.tobytes(),
        )

    def _infer_super_resolution(
        self,
        frame: np.ndarray,
        decision: SuperResolutionDecision,
        settings: LabSettings,
    ) -> tuple[Detection | None, CropRegion]:
        target = decision.target
        if target is None:
            raise ValueError("超解像の対象がありません")
        height, width = frame.shape[:2]
        region = expanded_crop_region(target, width, height)
        crop = frame[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ]
        enhanced_crop = self.super_resolution.enhance(crop)
        results = self.model.predict(
            enhanced_crop,
            conf=max(0.05, settings.confidence * 0.6),
            imgsz=max(320, min(settings.imgsz, 640)),
            device=self._resolved_device(),
            verbose=False,
        )
        local_detections = [
            item
            for item in self._extract_detections(results, settings)
            if item.label.casefold() == target.label.casefold()
        ]
        if not local_detections:
            return None, region
        mapped_detections = [
            remap_from_super_resolution(
                candidate,
                region,
                scale=self.super_resolution.scale,
                frame_width=width,
                frame_height=height,
            )
            for candidate in local_detections
        ]
        candidate = max(
            mapped_detections,
            key=lambda item: (
                intersection_over_union(target, item),
                item.confidence,
            ),
        )
        if intersection_over_union(target, candidate) < 0.05:
            return None, region
        return candidate, region

    @staticmethod
    def _merge_super_resolution_detection(
        detections: list[Detection],
        target: Detection | None,
        enhanced: Detection | None,
    ) -> tuple[list[Detection], bool]:
        if target is None or enhanced is None:
            return detections, False
        for index, item in enumerate(detections):
            if item is not target:
                continue
            if enhanced.confidence < item.confidence:
                return detections, False
            merged = list(detections)
            merged[index] = enhanced
            return merged, True
        return [*detections, enhanced], True

    def _maybe_auto_save(
        self,
        *,
        frame_number: int,
        detections: list[Detection],
        confirmed: list[Detection],
        settings: LabSettings,
        jpeg_payload: bytes,
    ) -> None:
        now = time.monotonic()
        with self.lock:
            if self.next_auto_save_at is None:
                self.next_auto_save_at = now + self.auto_save_interval_sec
                return
            if now < self.next_auto_save_at:
                return
            while self.next_auto_save_at <= now:
                self.next_auto_save_at += self.auto_save_interval_sec

            self.auto_save_count += 1
            save_number = self.auto_save_count
            width = self.frame_width
            height = self.frame_height

        stable_detection = smallest(confirmed)
        selected = stable_detection or smallest(detections)
        if stable_detection is not None:
            detection_status = "stable"
        elif selected is not None:
            detection_status = "unstable"
        else:
            detection_status = "not_detected"

        timestamp = datetime.now()
        image_path = self.output_dir / (
            f"auto_{save_number:04d}_{timestamp:%H%M%S}.jpg"
        )
        image_path.write_bytes(jpeg_payload)

        if selected is not None:
            data = selected.as_dict(width, height)
            detection_values: list[object] = [
                selected.label,
                "super_resolution" if selected.enhanced else "full_frame",
                selected.confidence,
                selected.x,
                selected.y,
                selected.width,
                selected.height,
                selected.area_px,
                data["area_pct"],
                selected.streak,
            ]
        else:
            detection_values = ["", "", "", "", "", "", "", "", "", 0]

        row = [
            timestamp.isoformat(timespec="milliseconds"),
            "auto",
            detection_status,
            frame_number,
            "",
            "",
            "",
            *detection_values,
            settings.target_classes,
            settings.imgsz,
            width,
            height,
            image_path.name,
        ]
        with self.measurements_lock:
            with self.measurements_path.open(
                "a", newline="", encoding="utf-8-sig"
            ) as stream:
                csv.writer(stream).writerow(row)
        with self.lock:
            self.last_auto_save_path = str(image_path.resolve())

    def _extract_detections(
        self, results: list[Any], settings: LabSettings
    ) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        class_filter = settings.class_filter()
        output: list[Detection] = []
        for box, confidence, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            if isinstance(names, dict):
                label = str(names.get(int(class_id), class_id))
            else:
                label = str(names[int(class_id)])
            if class_filter and label.casefold() not in class_filter:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in box.tolist())
            output.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    x=max(0, x1),
                    y=max(0, y1),
                    width=max(1, x2 - x1),
                    height=max(1, y2 - y1),
                )
            )
        return output

    def _append_detections(
        self,
        detections: list[Detection],
        frame_number: int,
        width: int,
        height: int,
        settings: LabSettings,
        inference_ms: float,
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        rows = []
        for detection in detections:
            data = detection.as_dict(width, height)
            rows.append(
                [
                    timestamp,
                    frame_number,
                    detection.label,
                    "super_resolution" if detection.enhanced else "full_frame",
                    detection.confidence,
                    detection.x,
                    detection.y,
                    detection.width,
                    detection.height,
                    detection.area_px,
                    data["width_pct"],
                    data["height_pct"],
                    data["area_pct"],
                    detection.streak,
                    settings.imgsz,
                    round(inference_ms, 2),
                    width,
                    height,
                ]
            )
        if rows:
            with self.csv_path.open("a", newline="", encoding="utf-8-sig") as stream:
                csv.writer(stream).writerows(rows)
