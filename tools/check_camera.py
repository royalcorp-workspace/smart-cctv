"""Camera Diagnostic Tool for TCP and RTSP Stream Validation."""

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlsplit

import cv2
import numpy as np

# Ensure workspace root is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from engine.config_loader import load_camera_config


def mask_url_credentials(url: str) -> str:
    """Mask password in URL for secure console display."""
    try:
        parts = urlsplit(url)
        if parts.password:
            masked_netloc = parts.netloc.replace(f":{parts.password}@", ":******@")
            return parts._replace(netloc=masked_netloc).geturl()
    except Exception:
        pass
    return url


def test_tcp_connectivity(host: str, port: int = 554, timeout_sec: float = 3.0) -> Tuple[bool, str]:
    """Test raw TCP socket connection to target host and port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_sec)
            sock.connect((host, port))
            return True, f"TCP Port {port} is OPEN and reachable on {host}."
    except socket.timeout:
        return False, f"Connection TIMED OUT after {timeout_sec}s connecting to {host}:{port}. Host might be offline."
    except ConnectionRefusedError:
        return False, f"Connection REFUSED by {host}:{port}. Camera device is reachable, but Port {port} is closed/disabled."
    except socket.gaierror as e:
        return False, f"DNS/Address resolution error for host '{host}': {e}."
    except OSError as e:
        return False, f"Socket error reaching {host}:{port}: {e}."


# Force TCP transport for RTSP streaming
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def test_stream_decode(source: Union[int, str]) -> Tuple[bool, Optional[Tuple[int, int]], float, str]:
    """Test opening stream and grabbing one valid decoded frame."""
    backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp://") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        return False, None, 0.0, "OpenCV VideoCapture failed to establish RTSP session (Handshake/Authentication failure)."

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()   
        return False, None, 0.0, "Connected to stream, but failed to decode video frame (Corrupted feed, invalid codec, or path error)."

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    return True, (w, h), fps, "Frame successfully grabbed and decoded."


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CCTV Camera Diagnostic Suite")
    parser.add_argument(
        "--cam",
        type=str,
        default="cam_01",
        help="Camera workspace identifier to test (default: cam_01)",
    )
    args = parser.parse_args()

    cam_dir = _WORKSPACE_DIR / "cameras" / args.cam
    config_file = cam_dir / "config.json"

    print("==================================================")
    print(f"      Smart CCTV 2.0 - Camera Diagnostic Tool     ")
    print("==================================================")
    print(f"[TARGET] Testing camera workspace: {args.cam}")

    if not config_file.exists():
        print(f"[FATAL] Config file missing: {config_file}")
        sys.exit(1)

    try:
        cfg = load_camera_config(config_file)
    except Exception as e:
        print(f"[FATAL] Failed to parse configuration: {e}")
        sys.exit(1)

    source = cfg.get("source", 0)
    print(f"[SOURCE] {mask_url_credentials(str(source))}")

    # Case 1: Local webcam or video file
    if isinstance(source, int) or (isinstance(source, str) and not source.startswith("rtsp://")):
        print("[INFO] Non-RTSP source detected. Testing direct OpenCV capture...")
        ok, res, fps, msg = test_stream_decode(source)
        if ok:
            print(f"[SUCCESS] Device/File ready! Resolution: {res[0]}x{res[1]} | FPS: {fps:.1f}")
            sys.exit(0)
        else:
            print(f"[ERROR] Capture failed: {msg}")
            sys.exit(1)

    # Case 2: RTSP Stream
    parsed = urlsplit(source)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 554

    # Check 1: TCP Port
    print(f"\n[CHECK 1/2] Testing TCP Socket Connection to {host}:{port} (timeout 3s)...")
    tcp_ok, tcp_msg = test_tcp_connectivity(host=host, port=port, timeout_sec=3.0)

    if not tcp_ok:
        print(f"[FAIL] {tcp_msg}")
        print("\n--- Diagnostic Advice ---")
        print("1. Check physical network cable and camera power status.")
        print(f"2. Verify IP '{host}' and ensure port {port} (RTSP) is enabled on camera web UI.")
        print("3. Check firewall / VLAN routing between this server and camera subnet.")
        sys.exit(1)

    print(f"[PASS] {tcp_msg}")

    # Check 2: RTSP Handshake & Decode
    print("\n[CHECK 2/2] Testing RTSP Handshake & Frame Decoding...")
    rtsp_ok, res, fps, rtsp_msg = test_stream_decode(source)

    if not rtsp_ok:
        print(f"[FAIL] {rtsp_msg}")
        print("\n--- Diagnostic Advice ---")
        print("1. Kredensial autentikasi mungkin salah (401 Unauthorized). Periksa CAM01_USER & CAM01_PASS di .env.")
        print(f"2. Stream path '{parsed.path}' mungkin tidak sesuai model kamera (misal: /Streaming/Channels/101, /live/ch0, /h264Preview_01_main).")
        print("3. Pastikan format kompresi video kamera diatur ke H.264 (bukan H.265 / HEVC tanpa hardware decoder).")
        sys.exit(1)

    print(f"[PASS] {rtsp_msg}")
    print("\n==================================================")
    print("           STREAM DIAGNOSTIC SUCCESS              ")
    print("==================================================")
    print(f"  Camera ID      : {args.cam}")
    print(f"  Resolution     : {res[0]} x {res[1]}")
    print(f"  Native FPS     : {fps:.1f}")
    print(f"  Authentication : VALID")
    print("==================================================\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
