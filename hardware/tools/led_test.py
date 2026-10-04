#!/usr/bin/env python3
"""Cycle the status LEDs on the flashed MPU6050 sketch.

Green D5, yellow D6, red D7. Stop Django first — only one process can open the port.

  python hardware/tools/led_test.py --port /dev/cu.usbserial-1430
"""

from __future__ import annotations

import argparse
import os
import sys
import time


_STEPS = (
    ("LG", "green", "D5"),
    ("LY", "yellow", "D6"),
    ("LR", "red", "D7"),
    ("L0", "off", "all"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cycle SAMS status LEDs (LG / LY / LR / L0)")
    parser.add_argument("--port", default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=1.5, help="How long each color stays on")
    args = parser.parse_args()

    port = args.port or os.environ.get("SAMS_HW_SERIAL_PORT", "").strip() or "/dev/cu.usbserial-1410"

    try:
        import serial
    except ImportError:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        return 1

    print(f"Opening {port} @ {args.baud} …")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.2)
    except Exception as exc:
        print(f"FAILED to open port: {exc}", file=sys.stderr)
        print("Stop Django runserver / Serial Monitor, then retry.", file=sys.stderr)
        return 1

    time.sleep(2.0)
    ser.reset_input_buffer()

    print("Watch the breadboard. Each color should light alone, then all go dark.")
    try:
        for cmd, name, pin in _STEPS:
            print(f"TX {cmd}  — {name} ({pin})")
            ser.write(f"{cmd}\n".encode("ascii"))
            ser.flush()
            time.sleep(args.seconds)
    finally:
        ser.close()

    print("Done. If none of them lit, check long leg → resistor → D5/D6/D7 and short leg → GND.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
