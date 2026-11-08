import csv
import json
import threading

import cv2
import numpy as np
import pytest

from small_object_lab.recording import RawRecorder


def read_recording(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return manifest, rows


def test_close_preserves_original_pixels_order_and_receipt_times(tmp_path):
    directory = tmp_path / "session" / "recording"
    recorder = RawRecorder(directory, metadata={"settings": {"target_classes": "cup"}})
    assert read_recording(directory)[0]["complete"] is False
    original = np.arange(24 * 36 * 3, dtype=np.uint8).reshape(24, 36, 3)
    frame = original.copy()
    assert recorder.submit(frame, timestamp_s=40.0) == 1
    frame[:] = 255  # An annotation / reused camera buffer must not alter the PNG.
    assert recorder.submit(frame, timestamp_s=40.125) == 2
    recorder.close()
    recorder.close()

    manifest, rows = read_recording(directory)
    assert manifest["schema_version"] == 1
    assert manifest["format"] == "png_sequence"
    assert manifest["complete"] is True
    assert manifest["errors"] == []
    assert manifest["dropped_frames"] == 0
    assert manifest["source_dimensions"] == [{"width": 36, "height": 24}]
    assert manifest["metadata"]["settings"]["target_classes"] == "cup"
    assert [int(row["frame_id"]) for row in rows] == [1, 2]
    assert [float(row["timestamp_s"]) for row in rows] == [0, 0.125]
    for row, expected in zip(rows, [original, frame]):
        saved = cv2.imdecode(
            np.frombuffer((directory / row["path"]).read_bytes(), dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(saved, expected)
    with pytest.raises(RuntimeError, match="closed"):
        recorder.submit(frame)


def test_full_queue_reports_drop_without_renumbering(monkeypatch, tmp_path):
    encode_started = threading.Event()
    second_started = threading.Event()
    release_encoder = threading.Event()
    original_encode = cv2.imencode
    call_count = 0

    def blocked_encoder(*args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            encode_started.set()
            assert release_encoder.wait(5)
        if call_count == 2:
            second_started.set()
        return original_encode(*args)

    monkeypatch.setattr(cv2, "imencode", blocked_encoder)
    recorder = RawRecorder(tmp_path / "recording", queue_size=1)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        assert recorder.submit(frame, 10) == 1
        assert encode_started.wait(5)
        assert recorder.submit(frame, 11) == 2
        assert recorder.submit(frame, 12) == 3
        assert recorder.status()["dropped_frames"] == 1
        release_encoder.set()
        assert second_started.wait(5)
        assert recorder.submit(frame, 13) == 4
    finally:
        release_encoder.set()
        recorder.close()
    manifest, rows = read_recording(recorder.output_dir)
    assert [int(row["frame_id"]) for row in rows] == [1, 2, 4]
    assert manifest["submitted_frames"] == 4
    assert manifest["saved_frames"] == 3
    assert manifest["dropped_frames"] == manifest["queue_drops"] == 1
    assert manifest["complete"] is True  # Closed correctly, but with documented gaps.


def test_size_limit_stops_recording_without_exceeding_budget(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    recorder = RawRecorder(tmp_path / "recording", max_bytes=len(encoded))
    recorder.submit(frame, 1)
    recorder.submit(frame, 2)
    recorder.close()
    manifest, rows = read_recording(recorder.output_dir)
    assert len(rows) == 1
    assert manifest["state"] == "limit_reached"
    assert manifest["bytes_written"] <= len(encoded)
    assert manifest["dropped_frames"] == 1
    assert recorder.status()["recording"] is False


def test_write_failure_is_visible_and_accounts_for_pending_frames(monkeypatch, tmp_path):
    def broken_encoder(*args):
        raise OSError("simulated disk/encoder failure")

    monkeypatch.setattr(cv2, "imencode", broken_encoder)
    recorder = RawRecorder(tmp_path / "recording")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(3):
        recorder.submit(frame, index)
    recorder.close()
    manifest, rows = read_recording(recorder.output_dir)
    assert rows == []
    assert manifest["state"] == "error"
    assert manifest["complete"] is False
    assert manifest["dropped_frames"] == 3
    assert "simulated" in manifest["errors"][0]["message"]


def test_existing_session_is_never_overwritten_and_times_are_monotonic(tmp_path):
    recorder = RawRecorder(tmp_path / "recording")
    with pytest.raises(FileExistsError):
        RawRecorder(tmp_path / "recording")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    recorder.submit(frame, 100)
    with pytest.raises(ValueError, match="backwards"):
        recorder.submit(frame, 99)
    recorder.close()


def test_lab_records_only_between_start_and_stop(tmp_path):
    from small_object_lab.detector import SmallObjectLab

    lab = SmallObjectLab(device="cpu", output_root=tmp_path)
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    lab.submit_frame(frame)  # Before the start button: inference only.
    assert lab.status()["recording"] == {"active": False, "count": 0, "last": None}
    assert lab.mailbox.wait_for_next(0)[2] == ("", None)
    with pytest.raises(ValueError, match="記録していません"):
        lab.stop_recording()

    first = lab.start_recording()["session_id"]
    with pytest.raises(ValueError, match="すでに記録中"):
        lab.start_recording()
    lab.update_settings({"target_classes": "cup"})
    lab.submit_frame(frame)
    lab.submit_frame(frame)
    assert lab.processed_frames == 0
    # Inference logs the raw frame it saw, numbered within this recording.
    assert lab.mailbox.wait_for_next(0)[2] == (first, 2)
    stopped = lab.stop_recording()
    assert stopped["complete"] is True and stopped["saved_frames"] == 2
    lab.submit_frame(frame)  # Between recordings: not saved.

    second = lab.start_recording()["session_id"]
    assert second != first
    lab.submit_frame(frame)
    assert lab.status()["recording"]["active"] is True
    lab.stop()  # Quitting the app finalizes an active recording.
    lab.submit_frame(frame)  # A blocked camera read returning after stop is ignored.

    manifest, rows = read_recording(lab.recordings_dir / first)
    assert manifest["session_id"] == first
    assert [int(row["frame_id"]) for row in rows] == [1, 2]
    assert manifest["metadata"]["settings_history"][0]["settings"]["target_classes"] == "cup"
    assert manifest["complete"] is True
    manifest, rows = read_recording(lab.recordings_dir / second)
    assert manifest["session_id"] == second
    assert [int(row["frame_id"]) for row in rows] == [1]
    assert manifest["complete"] is True
    assert lab.status()["recording"]["count"] == 2
