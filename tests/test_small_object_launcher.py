from small_object_lab import launcher
from small_object_lab.launcher import (
    AndroidPeer,
    find_online_android_peers,
    find_self_tailscale_ipv4,
)


def test_find_self_tailscale_ipv4_for_phone_remote_control():
    status = {
        "Self": {
            "OS": "windows",
            "TailscaleIPs": ["fd7a:115c:a1e0::1", "100.70.1.2"],
        }
    }

    assert find_self_tailscale_ipv4(status) == "100.70.1.2"
    assert find_self_tailscale_ipv4({}) is None


def test_find_online_android_peer_uses_tailscale_ipv4():
    status = {
        "Peer": {
            "node-key": {
                "OS": "android",
                "Online": True,
                "HostName": "camera-phone",
                "TailscaleIPs": [
                    "100.64.0.10",
                    "fd7a:115c:a1e0::ef36:b766",
                ],
            },
            "windows-key": {
                "OS": "windows",
                "Online": True,
                "HostName": "other-pc",
                "TailscaleIPs": ["100.1.2.3"],
            },
        }
    }

    assert find_online_android_peers(status) == [
        AndroidPeer(name="camera-phone", ip="100.64.0.10")
    ]


def test_find_online_android_peer_ignores_offline_phone():
    status = {
        "Peer": {
            "node-key": {
                "OS": "android",
                "Online": False,
                "HostName": "camera-phone",
                "TailscaleIPs": ["100.64.0.10"],
            }
        }
    }

    assert find_online_android_peers(status) == []


def test_discover_android_source_builds_android_camera_streamer_mjpeg_url(
    monkeypatch,
):
    monkeypatch.setattr(
        launcher,
        "read_tailscale_status",
        lambda: {
            "Peer": {
                "node-key": {
                    "OS": "android",
                    "Online": True,
                    "HostName": "camera-phone",
                    "TailscaleIPs": ["100.64.0.10"],
                }
            }
        },
    )

    peer, source = launcher.discover_android_source()

    assert peer.ip == "100.64.0.10"
    assert source == "http://100.64.0.10:8080/video"
