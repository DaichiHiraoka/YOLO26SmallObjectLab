from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .detector import SmallObjectLab
from .web import DASHBOARD_HTML, REMOTE_HTML


class LabHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], lab: SmallObjectLab) -> None:
        super().__init__(address, LabRequestHandler)
        self.lab = lab


class LabRequestHandler(BaseHTTPRequestHandler):
    server: LabHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._html(DASHBOARD_HTML)
        elif path == "/remote":
            self._html(REMOTE_HTML)
        elif path == "/api/status":
            self._json(HTTPStatus.OK, self.server.lab.status())
        elif path == "/api/stream":
            self._mjpeg_stream()
        elif path == "/api/detections.csv":
            self._file(self.server.lab.csv_path, "text/csv; charset=utf-8")
        elif path == "/api/frames.csv":
            self._file(
                self.server.lab.frames_path, "text/csv; charset=utf-8"
            )
        elif path == "/api/measurements.csv":
            self._file(
                self.server.lab.measurements_path, "text/csv; charset=utf-8"
            )
        elif path == "/health":
            self._json(HTTPStatus.OK, {"ok": True})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";")[0].strip().casefold() != "application/json":
            # Browsers must send a CORS preflight for this type, so other
            # websites cannot start or stop recordings through a plain form.
            self._json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Content-Type: application/json が必要です"},
            )
            return
        try:
            values = self._read_json()
            if path == "/api/recording/start":
                recording = self.server.lab.start_recording()
                self._json(HTTPStatus.OK, {"ok": True, "recording": recording})
            elif path == "/api/recording/stop":
                recording = self.server.lab.stop_recording()
                self._json(HTTPStatus.OK, {"ok": True, "recording": recording})
            elif path == "/api/settings":
                settings = self.server.lab.update_settings(values)
                self._json(HTTPStatus.OK, {"ok": True, "settings": vars(settings)})
            elif path == "/api/reset":
                self.server.lab.reset_record()
                self._json(HTTPStatus.OK, {"ok": True})
            elif path == "/api/snapshot":
                saved = self.server.lab.snapshot()
                self._json(HTTPStatus.OK, {"ok": True, "path": str(saved)})
            elif path == "/api/record":
                saved = self.server.lab.record_measurement(values)
                self._json(HTTPStatus.OK, {"ok": True, "path": str(saved)})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)}
            )

    def _read_json(self) -> dict[str, object]:
        length = self._content_length()
        if length == 0:
            return {}
        if length > 64 * 1024:
            raise ValueError("JSONが大きすぎます")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON objectを指定してください")
        return data

    def _content_length(self) -> int:
        try:
            return int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return 0

    def _mjpeg_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        previous_payload: bytes | None = None
        try:
            while True:
                with self.server.lab.lock:
                    payload = self.server.lab.latest_jpeg
                if payload is not None and payload is not previous_payload:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(payload)}\r\n\r\n".encode()
                    )
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    previous_payload = payload
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _html(self, content: str) -> None:
        payload = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: HTTPStatus, content: dict[str, object]) -> None:
        payload = json.dumps(content, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _file(self, path: Path, content_type: str) -> None:
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, message: str, *args: object) -> None:
        print(f"[HTTP] {self.address_string()} {message % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tailscale経由のスマホカメラでYOLO26小物体検出を試します。"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--source",
        required=True,
        help="android-camera-streamerのMJPEG URL。例: http://100.x.y.z:8080/video",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, 0 などUltralyticsのdevice指定",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/yolo26_small_object_lab"),
    )
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--remote-host",
        help="スマホ操作用に追加で待ち受けるPCのTailscale IP。例: 100.x.y.z",
    )
    return parser


def run_server(
    source: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    model: str = "yolo26n.pt",
    device: str = "auto",
    output_dir: Path = Path("outputs/yolo26_small_object_lab"),
    open_browser: bool = False,
    remote_host: str | None = None,
) -> int:
    lab = SmallObjectLab(
        model_name=model,
        device=device,
        source=source,
        output_root=output_dir,
    )
    server = LabHTTPServer((host, port), lab)
    servers = [server]
    if remote_host:
        # Listen on the Tailscale address only, never on every LAN interface.
        try:
            servers.append(LabHTTPServer((remote_host, port), lab))
            lab.remote_url = f"http://{remote_host}:{port}/remote"
        except OSError as error:
            print(f"スマホ操作用の待受を開始できません ({remote_host}:{port}): {error}")
    for extra in servers[1:]:
        threading.Thread(
            target=extra.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
        ).start()
    lab.start()

    stopping = threading.Event()

    def stop_server(*_: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        for item in servers:
            threading.Thread(target=item.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_server)

    local_url = f"http://{host}:{port}"
    print(f"YOLO26 Small Object Lab: {local_url}")
    if lab.remote_url:
        print(f"スマホから記録開始・終了: {lab.remote_url}")
    print(f"Camera: {source}")
    print(f"Output: {lab.output_dir.resolve()}")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(local_url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_server()
        for item in servers:
            item.server_close()
        lab.stop()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run_server(
        source=args.source,
        host=args.host,
        port=args.port,
        model=args.model,
        device=args.device,
        output_dir=args.output_dir,
        open_browser=args.open_browser,
        remote_host=args.remote_host,
    )


if __name__ == "__main__":
    raise SystemExit(main())
