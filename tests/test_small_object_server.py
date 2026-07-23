import json
import threading
import urllib.request

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
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
