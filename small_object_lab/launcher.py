from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AndroidPeer:
    name: str
    ip: str


def _iter_peer_values(peer_data: object) -> list[dict[str, Any]]:
    if isinstance(peer_data, dict):
        return [value for value in peer_data.values() if isinstance(value, dict)]
    if isinstance(peer_data, list):
        return [value for value in peer_data if isinstance(value, dict)]
    return []


def _tailscale_ipv4(node: dict[str, Any]) -> str | None:
    addresses = node.get("TailscaleIPs")
    if not isinstance(addresses, list):
        return None
    return next(
        (
            str(address)
            for address in addresses
            if isinstance(address, str)
            and address.startswith("100.")
            and ":" not in address
        ),
        None,
    )


def find_self_tailscale_ipv4(status: dict[str, Any]) -> str | None:
    node = status.get("Self")
    return _tailscale_ipv4(node) if isinstance(node, dict) else None


def find_online_android_peers(status: dict[str, Any]) -> list[AndroidPeer]:
    peers: list[AndroidPeer] = []
    for peer in _iter_peer_values(status.get("Peer")):
        if str(peer.get("OS", "")).casefold() != "android":
            continue
        if not bool(peer.get("Online")):
            continue
        ipv4 = _tailscale_ipv4(peer)
        if ipv4:
            peers.append(
                AndroidPeer(
                    name=str(peer.get("HostName") or peer.get("DNSName") or "Android"),
                    ip=ipv4,
                )
            )
    return peers


def read_tailscale_status() -> dict[str, Any]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            ["tailscale", "status", "--json"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            creationflags=creation_flags,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "PCにTailscaleが見つかりません。Tailscaleを起動してから再実行してください。"
        ) from error
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", "") or ""
        raise RuntimeError(
            "PCのTailscaleへ接続できません。Tailscaleを起動してから再実行してください。"
            f"\n{detail.strip()}"
        ) from error

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Tailscaleの端末情報を読み取れませんでした。") from error
    if not isinstance(data, dict) or data.get("BackendState") != "Running":
        raise RuntimeError("PCのTailscaleが接続状態ではありません。")
    return data


def discover_android_source(
    port: int = 8080, status: dict[str, Any] | None = None
) -> tuple[AndroidPeer, str]:
    if status is None:
        status = read_tailscale_status()
    peers = find_online_android_peers(status)
    if not peers:
        raise RuntimeError(
            "オンラインのAndroid端末がTailscale上に見つかりません。\n"
            "AndroidでTailscaleとandroid-camera-streamerを起動し、"
            "streamerのIPにTailscaleの100.xアドレスを選んでください。"
        )
    if len(peers) > 1:
        names = ", ".join(f"{peer.name} ({peer.ip})" for peer in peers)
        raise RuntimeError(
            "オンラインのAndroid端末が複数あります。1台だけ接続してください。\n"
            f"検出端末: {names}"
        )
    peer = peers[0]
    return peer, f"http://{peer.ip}:{port}/video"


def show_error(message: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("YOLO26 Small Object Lab", message)
        root.destroy()
    except Exception:
        print(f"エラー: {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Android camera over Tailscale -> YOLO26 small-object lab")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO weights for the live view, e.g. the key detector "
        "outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt",
    )
    parser.add_argument(
        "--sr-model",
        default="espcn",
        choices=["espcn", "fsrcnn", "lapsrn", "edsr"],
        help="Super-resolution model for local re-detection (default: espcn)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.model != "yolo26n.pt" and not Path(args.model).is_file():
            # Ultralytics would try to download an unknown name; fail clearly instead.
            raise RuntimeError(f"モデルファイルが見つかりません: {args.model}")
        status = read_tailscale_status()
        peer, source = discover_android_source(status=status)
        print(f"Android: {peer.name} ({peer.ip})")
        print(f"android-camera-streamer MJPEG: {source}")

        from .server import run_server

        return run_server(
            source=source,
            model=args.model,
            sr_model=args.sr_model,
            open_browser=True,
            remote_host=find_self_tailscale_ipv4(status),
        )
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        show_error(str(error))
        return 1
