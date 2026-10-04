"""Open the board serial link when the server starts."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger("hardware.bringup")

ROOT = Path(__file__).resolve().parents[1]


def find_serial_port() -> str:
    """Use SAMS_HW_SERIAL_PORT, or the first USB serial device on this machine."""
    chosen = os.environ.get("SAMS_HW_SERIAL_PORT", "").strip()
    if chosen:
        return chosen
    found = sorted(
        glob.glob("/dev/cu.usbserial-*")
        + glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/cu.wchusbserial*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
    )
    return found[0] if found else ""


def ensure_certs() -> tuple[Path, Path]:
    cert = ROOT / "hardware" / "certs" / "cert.pem"
    key = ROOT / "hardware" / "certs" / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    script = ROOT / "hardware" / "tools" / "gen_certs.sh"
    subprocess.run([str(script)], check=True)
    return cert, key


def prepare_environment() -> str:
    """Set the serial port and HTTPS flag before Django loads settings users."""
    port = find_serial_port()
    if port:
        os.environ["SAMS_HW_SERIAL_PORT"] = port
    os.environ.setdefault("SAMS_HTTPS", "1")
    return port


def start_hardware() -> None:
    """Connect the accelerometer, distance sensor, buzzer, and lights. Keep serving if the board is absent."""
    from .config import load_config
    from .presence import get_presence
    from .stream import get_stream

    port = os.environ.get("SAMS_HW_SERIAL_PORT", "").strip()
    stream = get_stream()
    if not port:
        get_presence().reset()
        print("No USB serial port. Accelerometer, distance sensor, and buzzer stay disconnected.", flush=True)
        return
    stream.config = replace(load_config(), serial_port=port)
    try:
        stream.set_mode("live")
    except Exception as exc:  # noqa: BLE001
        get_presence().reset()
        logger.warning("Hardware serial did not open: %s", exc)
        print(f"Hardware serial not connected ({port}): {exc}", flush=True)
        return
    print(f"Hardware serial open on {port}", flush=True)
