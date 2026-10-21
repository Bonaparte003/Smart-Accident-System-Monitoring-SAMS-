#!/usr/bin/env python3
"""One-shot check of every SAMS actuator and sensor over serial (no Django).

Stop `runserver` first — only one process can open the port.

Usage:
  python hardware/tools/hw_check.py --port /dev/cu.usbserial-1410
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time


def _rx(ser) -> str:
    return ser.readline().decode("utf-8", errors="ignore").strip()


def _parse_accel(line: str):
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def _wait_ack(ser, cmd: str, timeout: float = 3.0) -> bool:
    ser.write(f"{cmd}\n".encode())
    ser.flush()
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = _rx(ser)
        if f"ACK {cmd}" in line:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAMS hardware end to end")
    parser.add_argument("--port", default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to sample the MPU6050")
    args = parser.parse_args()
    port = args.port or os.environ.get("SAMS_HW_SERIAL_PORT", "").strip() or "/dev/cu.usbserial-1410"

    try:
        import serial
    except ImportError:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        return 1

    print(f"Opening {port} @ {args.baud} (the board resets — watch for the startup sequence)")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.3)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL port: {exc}\n  Stop Django / serial monitors, check the USB cable.", file=sys.stderr)
        return 1

    results: dict[str, bool] = {}

    banner = False
    boot_notes: list[str] = []
    deadline = time.time() + 6.0
    while time.time() < deadline and not banner:
        line = _rx(ser)
        if line.startswith("#"):
            boot_notes.append(line)
            banner = banner or "SAMS ready" in line
    results["firmware banner"] = banner
    for note in boot_notes:
        print(f"  RX {note}")

    print(f"\nSampling the MPU6050 for {args.seconds:.0f}s — tap or shake the sensor now")
    samples = []
    tof = []
    notes: list[str] = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        line = _rx(ser)
        if not line:
            continue
        if line.startswith("TOF:"):
            try:
                tof.append(float(line[4:]))
            except ValueError:
                pass
        elif line.startswith("#"):
            notes.append(line)
        else:
            xyz = _parse_accel(line)
            if xyz:
                samples.append(xyz)
    elapsed = time.time() - t0

    rate = len(samples) / elapsed
    print(f"  {len(samples)} samples → {rate:.1f} Hz (want ~50)")
    if samples:
        mags = [(x * x + y * y + z * z) ** 0.5 for x, y, z in samples]
        mean_g = statistics.fmean(mags)
        spread = statistics.pstdev(mags)
        distinct = len(set(samples))
        print(f"  |a| mean {mean_g:.3f} g (want ~1.0), std {spread:.4f} g, {distinct} distinct readings")
        print(f"  last sample {samples[-1]}")
        results["MPU6050 streaming ~50 Hz"] = 40 <= rate <= 60
        results["MPU6050 reads ~1 g"] = 0.8 <= mean_g <= 1.2
        results["MPU6050 values change"] = distinct > 1
        if spread < 0.01:
            print("  ! values barely moved — did you shake it? A frozen bus repeats one reading.")
    else:
        results["MPU6050 streaming ~50 Hz"] = False
    for note in sorted(set(notes)):
        print(f"  RX {note}")

    if tof:
        print(f"\n  ToF: {len(tof)} readings, last {tof[-1]:.2f} m")
    else:
        print("\n  ToF: no in-range readings (nothing within 0.02–4 m, or sensor missing)")
    results["ToF readings"] = bool(tof)

    print("\nActuators — watch and listen")
    print("  B1 buzzer on (1 s)")
    results["buzzer ACK B1"] = _wait_ack(ser, "B1")
    time.sleep(1.0)
    results["buzzer ACK B0"] = _wait_ack(ser, "B0")
    for cmd, name in (("LG", "green"), ("LY", "yellow"), ("LR", "red")):
        print(f"  {cmd} {name} LED")
        ser.write(f"{cmd}\n".encode())
        time.sleep(0.7)
    ser.write(b"L0\n")
    for cmd, name in (("M120", "low"), ("M180", "medium"), ("M255", "high")):
        print(f"  {cmd} machine motor {name} (1.5 s)")
        results[f"machine motor ACK {cmd}"] = _wait_ack(ser, cmd)
        time.sleep(1.5)
    ser.write(b"V0\n")
    print("  ST startup sequence (buzzer → green → yellow → red → motor)")
    results["startup sequence ACK ST"] = _wait_ack(ser, "ST", timeout=5.0)
    time.sleep(0.5)
    ser.close()

    print("\nSummary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("  (LEDs and motor have no ACK — confirm by eye)")
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
