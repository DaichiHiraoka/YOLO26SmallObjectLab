import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from small_object_lab.detector import SmallObjectLab
from small_object_lab.server import LabHTTPServer


def request(url: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    method = "GET" if data is None else "POST"
    with urllib.request.urlopen(
        urllib.request.Request(
            url + path, data=data, headers=headers, method=method
        ),
        timeout=15,
    ) as response:
        return response.status, response.read()


def test_server_exposes_dashboard_and_updates_settings(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path)
    server = LabHTTPServer(("127.0.0.1", 0), lab)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, dashboard = request(base_url, "/")
        assert status == 200
        assert "YOLO26" in dashboard.decode()
        assert "15秒ごと" in dashboard.decode()
        assert "フレームCSV" in dashboard.decode()

        status, frame_csv = request(base_url, "/api/frames.csv")
        assert status == 200
        assert b"sr_scales_tried" in frame_csv

        status, response = request(base_url, "/api/status")
        assert status == 200
        auto_save = json.loads(response)
        assert auto_save["auto_save_interval_sec"] == 15
        assert auto_save["auto_save_count"] == 0

        status, response = request(
            base_url,
            "/api/settings",
            {
                "confidence": 0.15,
                "imgsz": 1280,
                "stable_frames": 5,
                "target_classes": "cell phone",
                "super_resolution_enabled": True,
                "sr_trigger_confidence": 0.4,
                "sr_trigger_area_pct": 0.75,
                "sr_target_min_px": 128,
                "sr_max_scale": 8,
            },
        )
        assert status == 200
        settings = json.loads(response)["settings"]
        assert settings == {
            "confidence": 0.15,
            "imgsz": 1280,
            "target_classes": "cell phone",
            "stable_frames": 5,
            "super_resolution_enabled": True,
            "sr_trigger_confidence": 0.4,
            "sr_trigger_area_pct": 0.75,
            "sr_target_min_px": 128,
            "sr_max_scale": 8,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_recording_is_controlled_from_dashboard_and_phone_page(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path)
    server = LabHTTPServer(("127.0.0.1", 0), lab)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, dashboard = request(base_url, "/")
        assert "記録開始" in dashboard.decode()
        status, phone = request(base_url, "/remote")
        assert status == 200
        assert "記録開始" in phone.decode() and "記録終了" in phone.decode()

        status, response = request(base_url, "/api/recording/start", {})
        recording_id = json.loads(response)["recording"]["session_id"]
        lab.submit_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        status, response = request(base_url, "/api/status")
        assert json.loads(response)["recording"]["active"] is True

        status, response = request(base_url, "/api/recording/stop", {})
        stopped = json.loads(response)["recording"]
        assert stopped["session_id"] == recording_id
        assert stopped["complete"] is True and stopped["saved_frames"] == 1
        assert (lab.recordings_dir / recording_id / "manifest.json").exists()

        with pytest.raises(urllib.error.HTTPError) as error:
            request(base_url, "/api/recording/stop", {})
        assert error.value.code == 400

        # A cross-site form post cannot send application/json.
        form = urllib.request.Request(
            base_url + "/api/recording/start",
            data=b"{}",
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(form, timeout=15)
        assert error.value.code == 415
        assert lab.status()["recording"]["active"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        lab.stop()


def test_phone_can_watch_the_annotated_stream_at_a_lower_frame_rate(tmp_path):
    from small_object_lab.server import stream_fps

    assert stream_fps(None) is None and stream_fps("10") == 10.0
    for bad in ("0", "31", "x"):
        with pytest.raises(ValueError):
            stream_fps(bad)

    lab = SmallObjectLab(output_root=tmp_path)
    lab.latest_jpeg = b"\xff\xd8annotated\xff\xd9"
    server = LabHTTPServer(("127.0.0.1", 0), lab)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, phone = request(base_url, "/remote")
        assert "検出結果を表示" in phone.decode() and "/api/stream?fps=10" in phone.decode()
        with urllib.request.urlopen(base_url + "/api/stream?fps=10", timeout=15) as stream:
            assert stream.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            part = stream.read(72)  # exactly one part; the frame never changes
        assert b"Content-Length: 13\r\n\r\n\xff\xd8annotated\xff\xd9" in part
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base_url, "/api/stream?fps=0")
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        lab.stop()


def test_dashboard_defaults_to_cup_and_works_from_the_phone(tmp_path):
    lab = SmallObjectLab(output_root=tmp_path)
    assert lab.settings.target_classes == "cup" and lab.settings.class_filter() == {"cup"}
    server = LabHTTPServer(("127.0.0.1", 0), lab)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, dashboard = request(f"http://127.0.0.1:{server.server_port}", "/")
        page = dashboard.decode()
        # full rate on the PC, 10 fps when the same page is opened over Tailscale
        assert 'value="cup"' in page and '"/api/stream?fps=10"' in page
        status, response = request(f"http://127.0.0.1:{server.server_port}", "/api/status")
        assert json.loads(response)["settings"]["target_classes"] == "cup"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        lab.stop()
