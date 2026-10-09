#!/usr/bin/env python3
"""Standalone buzzer serial test (no Django).

Stop `runserver` first — only one process can open the port.

Usage:
  python hardware/tools/buzzer_test.py
  python hardware/tools/buzzer_test.py --port /dev/cu.usbserial-1410
"""

from __future__ import annotations

import argparse
import sys
import time


def _buzzer_off(ser, attempts: int = 5) -> bool:
    """Send B0 until the board ACKs it. Returns True once confirmed."""
    print("TX B0 (off)")
    for _ in range(attempts):
        ser.write(b"B0\n")
        ser.flush()
        deadline = time.time() + 0.4
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                print(f"  RX {line}")
            if "ACK B0" in line:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Send B1/B0 to SAMS Arduino buzzer")
    parser.add_argument(
        "--port",
        default="",
        help="Serial port (default: SAMS_HW_SERIAL_PORT or /dev/cu.usbserial-1410)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=2.0, help="How long to hold B1")
    args = parser.parse_args()

    import os

    port = args.port or os.environ.get("SAMS_HW_SERIAL_PORT", "").strip() or "/dev/cu.usbserial-1410"

    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        return 1

    print("Available ports:")
    for p in list_ports.comports():
        print(f"  {p.device}  {p.description}")

    print(f"\nOpening {port} @ {args.baud} …")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.3)
    except Exception as exc:
        print(f"FAILED to open port: {exc}", file=sys.stderr)
        print("Stop Django runserver / Serial Monitor, then retry.", file=sys.stderr)
        return 1

    time.sleep(2.0)  # allow MCU reset after open
    ser.reset_input_buffer()

    print("Listening for board banner (2s)…")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  RX {line}")

    got_ack = False
    off_acked = False
    try:
        print(f"\nTX B1 (on for {args.seconds}s)")
        for _ in range(5):
            ser.write(b"B1\n")
            ser.flush()
            time.sleep(0.05)

        t0 = time.time()
        while time.time() - t0 < args.seconds:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  RX {line}")
                if "ACK" in line and "B1" in line:
                    got_ack = True
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        off_acked = _buzzer_off(ser)
        ser.close()

    print()
    if off_acked:
        print("Buzzer OFF confirmed (ACK B0).")
    else:
        print("WARNING: board did not confirm B0 — buzzer may still be on. Unplug USB to silence it.")
    if got_ack:
        print("RESULT: Board ACK'd B1 — firmware received the command.")
        print("  If still silent: D8 cannot power this buzzer → use transistor/5V rail.")
        print("  Did onboard LED (D13) light with buzzer_d8_debug.ino? That proves pin toggle.")
    else:
        print("RESULT: No ACK from board.")
        print("  → Flash hardware/firmware/buzzer_d8_debug.ino")
        print("  → Confirm port, close Django, retry")
        print("  → If boot beeps missing after flash: wiring/current (not software)")
    return 0 if got_ack else 2


if __name__ == "__main__":
    raise SystemExit(main())
