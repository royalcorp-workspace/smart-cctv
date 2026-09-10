#!/usr/bin/env python3
"""
Multi-Camera Diagnostic Suite — Smart CCTV v2
=============================================
Menguji konektivitas jaringan TCP Port 554 dan RTSP H.264 frame decode handshake
secara sekuensial untuk seluruh workspace kamera di direktori `cameras/`.
"""

import argparse
import os
from pathlib import Path
import socket
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

import cv2
import numpy as np

# Force TCP transport for RTSP streaming (prevents UDP packet loss)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Bootstrap project root
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
            return True, f"Port {port} OPEN"
    except socket.timeout:
        return False, f"TIMED OUT ({timeout_sec}s)"
    except ConnectionRefusedError:
        return False, f"Port {port} REFUSED"
    except socket.gaierror as e:
        return False, f"DNS/Host Err: {e}"
    except OSError as e:
        return False, f"Socket Err: {e}"


def test_rtsp_decode(source: Union[int, str], timeout_sec: float = 5.0) -> Tuple[bool, Optional[Tuple[int, int]], float, float, str]:
    """Test opening RTSP stream via OpenCV FFmpeg TCP and decoding 1 valid frame.

    Returns:
        (success: bool, resolution: (w, h), fps: float, elapsed_sec: float, message: str)
    """
    t0 = time.time()
    backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp://") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)

    if not cap.isOpened():
        cap.release()
        elapsed = time.time() - t0
        return False, None, 0.0, elapsed, "RTSP Handshake / Auth GAGAL"

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    elapsed = time.time() - t0

    if not ret or frame is None:
        cap.release()
        return False, None, 0.0, elapsed, "Connected, tapi gagal decode frame"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    return True, (w, h), fps, elapsed, "Frame decode SUKSES"


def parse_rtsp_target(source_url: str) -> Tuple[str, int]:
    """Extract host and port from RTSP URL string."""
    try:
        parts = urlsplit(source_url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 554
        return host, port
    except Exception:
        return "127.0.0.1", 554


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Camera Network & RTSP Diagnostic Tool")
    parser.add_argument(
        "--cams",
        nargs="*",
        default=None,
        help="Spesifik kamera yang ingin diuji (contoh: --cams cam_01 cam_02 cam_03). Default: semua di cameras/",
    )
    args = parser.parse_args()

    cam_root = _WORKSPACE_DIR / "cameras"
    if not cam_root.exists():
        print(f"[FATAL] Direktori cameras/ tidak ditemukan di: {cam_root}")
        sys.exit(1)

    all_cam_dirs = [p for p in sorted(cam_root.iterdir()) if p.is_dir() and (p / "config.json").exists()]
    if args.cams:
        target_dirs = [p for p in all_cam_dirs if p.name in args.cams]
    else:
        target_dirs = all_cam_dirs

    if not target_dirs:
        print("[FATAL] Tidak ada workspace kamera yang ditemukan untuk diuji.")
        sys.exit(1)

    print("\n" + "=" * 76)
    print("        SMART CCTV 2.0 — MULTI-CAMERA DIAGNOSTIC SUITE (v2)        ")
    print("=" * 76)
    print(f"Total kamera yang akan diuji: {len(target_dirs)} ({', '.join(d.name for d in target_dirs)})\n")

    results = []

    for cam_dir in target_dirs:
        cam_id = cam_dir.name
        cfg_file = cam_dir / "config.json"
        print(f"[{cam_id}] Membaca konfigurasi: {cfg_file.name} ...")

        try:
            cfg = load_camera_config(cfg_file)
            cam_name = cfg.get("name", cam_id)
            source = cfg.get("source", "")
            masked_source = mask_url_credentials(source)
            print(f"       Nama    : {cam_name}")
            print(f"       Source  : {masked_source}")

            if isinstance(source, str) and source.startswith("rtsp://"):
                host, port = parse_rtsp_target(source)
                print(f"       [1/2] Socket Ping TCP {host}:{port} ... ", end="", flush=True)
                tcp_ok, tcp_msg = test_tcp_connectivity(host, port, timeout_sec=3.0)
                print(f"{'OK' if tcp_ok else 'GAGAL'} ({tcp_msg})")

                if tcp_ok:
                    print(f"       [2/2] RTSP Handshake & Frame Decode ... ", end="", flush=True)
                    decode_ok, res, fps, elapsed, dec_msg = test_rtsp_decode(source, timeout_sec=5.0)
                    if decode_ok:
                        res_str = f"{res[0]}x{res[1]}" if res else "Unknown"
                        print(f"OK ({res_str} @ {fps:.1f}fps in {elapsed:.2f}s)")
                        results.append({
                            "cam_id": cam_id,
                            "ip": host,
                            "port": port,
                            "tcp": "OPEN",
                            "decode": f"{res_str} @ {fps:.0f}fps",
                            "latency": f"{elapsed:.2f}s",
                            "status": "ONLINE",
                        })
                    else:
                        print(f"GAGAL ({dec_msg})")
                        results.append({
                            "cam_id": cam_id,
                            "ip": host,
                            "port": port,
                            "tcp": "OPEN",
                            "decode": "DECODE_ERR",
                            "latency": f"{elapsed:.2f}s",
                            "status": "FAILED_RTSP",
                        })
                else:
                    results.append({
                        "cam_id": cam_id,
                        "ip": host,
                        "port": port,
                        "tcp": "CLOSED",
                        "decode": "N/A",
                        "latency": "N/A",
                        "status": "UNREACHABLE",
                    })

            else:
                # Local video or test source
                print(f"       [Local File] Pengujian source lokal ... ", end="", flush=True)
                decode_ok, res, fps, elapsed, dec_msg = test_rtsp_decode(source, timeout_sec=3.0)
                res_str = f"{res[0]}x{res[1]}" if res else "Unknown"
                status_str = "ONLINE" if decode_ok else "FAILED"
                print(f"{status_str} ({res_str})")
                results.append({
                    "cam_id": cam_id,
                    "ip": "Local/File",
                    "port": 0,
                    "tcp": "N/A",
                    "decode": res_str,
                    "latency": f"{elapsed:.2f}s",
                    "status": status_str,
                })

        except Exception as e:
            print(f"       [ERROR] Gagal memproses kamera: {e}")
            results.append({
                "cam_id": cam_id,
                "ip": "Error",
                "port": 0,
                "tcp": "ERR",
                "decode": "ERR",
                "latency": "N/A",
                "status": "CONFIG_ERROR",
            })
        print()

    # Tampilkan Matriks Hasil
    print("=" * 76)
    print("                     MATRIKS HASIL DIAGNOSTIK                      ")
    print("=" * 76)
    header = f"{'KAMERA':<10} {'IP TARGET':<18} {'PORT':<6} {'TCP':<8} {'DECODE FRAME':<18} {'STATUS':<12}"
    print(header)
    print("-" * 76)
    for r in results:
        line = f"{r['cam_id']:<10} {r['ip']:<18} {r['port']:<6} {r['tcp']:<8} {r['decode']:<18} {r['status']:<12}"
        print(line)
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
