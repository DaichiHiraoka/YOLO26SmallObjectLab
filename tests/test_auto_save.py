import csv
import time

import pytest

from small_object_lab.detector import LabSettings, SmallObjectLab
from small_object_lab.metrics import Detection


def test_auto_save_records_stable_detection_and_image(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path, auto_save_interval_sec=15)
    lab.frame_width = 200
    lab.frame_height = 100
    lab.next_auto_save_at = time.monotonic() - 1
    settings = LabSettings(target_classes="cup", imgsz=1280, stable_frames=3)
    detection = Detection(
        label="cup",
        confidence=0.72,
        x=10,
        y=20,
        width=8,
        height=6,
        streak=3,
    )

    lab._maybe_auto_save(
        frame_number=42,
        detections=[detection],
        confirmed=[detection],
        settings=settings,
        jpeg_payload=b"test-jpeg",
    )

    with lab.measurements_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["mode"] == "auto"
    assert rows[0]["detection_status"] == "stable"
    assert rows[0]["label"] == "cup"
    assert rows[0]["x"] == "10"
    assert rows[0]["y"] == "20"
    assert rows[0]["frame"] == "42"
    assert rows[0]["image_file"].startswith("auto_0001_")
    assert (lab.output_dir / rows[0]["image_file"]).read_bytes() == b"test-jpeg"


def test_auto_save_records_not_detected(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path, auto_save_interval_sec=15)
    lab.frame_width = 200
    lab.frame_height = 100
    lab.next_auto_save_at = time.monotonic() - 1
    settings = LabSettings(target_classes="cup", imgsz=640)

    lab._maybe_auto_save(
        frame_number=7,
        detections=[],
        confirmed=[],
        settings=settings,
        jpeg_payload=b"empty-frame",
    )

    with lab.measurements_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["detection_status"] == "not_detected"
    assert rows[0]["label"] == ""
    assert rows[0]["image_file"].startswith("auto_0001_")


def test_auto_save_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="greater than 0"):
        SmallObjectLab(output_root=tmp_path, auto_save_interval_sec=0)
