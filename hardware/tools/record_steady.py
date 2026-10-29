#!/usr/bin/env python3
"""Record the MPU9250/6500 on a machine for the machine-state model (STOPPED or RUNNING data).

Mount the board on the machine the way it will stay in service, then record it
switched off (the default) or running (``--state RUNNING``). RUNNING recordings
from the same board and mounting let notebook 02 learn stopped vs running
rather than this board vs the UCI fan rig. Stop Django first: only one process
can open the port.

RUNNING drives the D9 motor at ``--motor`` (default medium) for the recording
and stops it afterwards; ``--motor none`` leaves a real machine you switch on
yourself. STOPPED sends the motor off first. Recording starts once the board
has booted, the motor has acknowledged, and it has had time to spin up.
Each run writes two files to ``stead_dataset/mpu6050/``:

- ``<time>[_name].csv``  sample, seconds_elapsed, x, y, z (g, as the firmware sends them)
- ``<time>[_name].json`` state, port, duration, sample count, and your notes

Notebook 02 (section 8) trains the machine-state model from every CSV there.
Leave-one-recording-out needs several takes per motor speed and several STOPPED
sessions (quiet / bumps / footsteps). Check progress with::

  ./hardware/tools/hw.sh coverage --next

Examples::

  python hardware/tools/record_steady.py --seconds 60 --name press01 --disturbance quiet --note "off, quiet"
  python hardware/tools/record_steady.py --seconds 60 --name press01 --disturbance bumps --note "off, bump mount"
  python hardware/tools/record_steady.py --state RUNNING --motor low --seconds 60 --name press01
  python hardware/tools/record_steady.py --state RUNNING --motor high --seconds 60 --name press01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE_HZ = 50.0
GRAVITY_RANGE_G = (0.8, 1.2)
RATE_TOLERANCE = 0.1
STATES = ("STOPPED", "RUNNING")
MOTOR_CMDS = {"low": "M120", "medium": "M180", "high": "M255", "intermittent": "V1", "none": ""}
MOTOR_OFF = "V0"
DISTURBANCES = ("quiet", "bumps", "footsteps")
BOOT_TIMEOUT_S = 6.0
ACK_TIMEOUT_S = 1.5
SETTLE_S = 1.5


def recording_problems(rows: list[tuple[int, float, float, float, float]], duration: float) -> list[str]:
    """Why this recording cannot be machine-state data, or an empty list.

    A board shaken by a running machine still averages about 1 g of gravity.
    """
    problems = []
    if len({(x, y, z) for _i, _t, x, y, z in rows}) == 1:
        problems.append("every sample is identical: the MPU9250/6500 is not answering (check SDA A2, SCL A3, VCC, GND)")
    mags = [math.sqrt(x * x + y * y + z * z) for _i, _t, x, y, z in rows]
    mean = sum(mags) / len(mags)
    if not GRAVITY_RANGE_G[0] <= mean <= GRAVITY_RANGE_G[1]:
        problems.append(f"mean |a| is {mean:.3f} g; a still board reads about 1 g")
    rate = measured_rate(rows, duration)
    if abs(rate - SAMPLE_RATE_HZ) > RATE_TOLERANCE * SAMPLE_RATE_HZ:
        problems.append(f"{rate:.1f} samples/s instead of {SAMPLE_RATE_HZ:.0f}: reflash the board (./hardware/tools/hw.sh flash)")
    return problems


def measured_rate(rows: list[tuple[int, float, float, float, float]], duration: float) -> float:
    """Samples per second between the first and last sample, so any gap before the first is not counted."""
    span = rows[-1][1] - rows[0][1] if len(rows) > 1 else 0.0
    return (len(rows) - 1) / span if span > 0 else len(rows) / duration


def read_line(ser) -> str:
    return ser.readline().decode("utf-8", errors="ignore").strip()


def wait_for_samples(ser, timeout: float) -> tuple[bool, str]:
    """Read until the board, which resets when the port opens, sends its first sample.

    Also returns the chip its boot banner names (``# SAMS ready chip=MPU6500 …``), or "unknown".
    """
    chip = "unknown"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(ser)
        if line.startswith("#"):
            print(f"  board: {line[1:].strip()}")
            if "chip=" in line:
                chip = line.split("chip=", 1)[1].split()[0]
        elif parse_sample(line) is not None:
            return True, chip
    return False, chip


def send_motor(ser, cmd: str) -> bool:
    """Send one motor command and wait for the firmware's ``# ACK``."""
    ser.write(f"{cmd}\n".encode("ascii"))
    deadline = time.monotonic() + ACK_TIMEOUT_S
    while time.monotonic() < deadline:
        if read_line(ser) == f"# ACK {cmd}":
            return True
    return False


def parse_sample(line: str) -> tuple[float, float, float] | None:
    """An ``ax,ay,az`` line from the firmware, or None for status and ToF lines."""
    if not line or line.startswith("#") or line.upper().startswith("TOF:"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Record STOPPED or RUNNING machine MPU9250/6500 windows for notebook 02")
    parser.add_argument("--state", type=str.upper, choices=STATES, default="STOPPED", help="What the machine is doing")
    parser.add_argument(
        "--motor", type=str.lower, choices=tuple(MOTOR_CMDS), default="medium",
        help="RUNNING only: D9 motor level for the recording, or none for a real machine you switch on yourself",
    )
    parser.add_argument(
        "--disturbance",
        type=str.lower,
        choices=DISTURBANCES,
        default="",
        help="STOPPED only: quiet / bumps / footsteps — kept in the JSON and file name for coverage checks",
    )
    parser.add_argument("--port", default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=60.0, help="Recording length (Ctrl+C stops early and keeps what was read)")
    parser.add_argument("--name", default="", help="Short tag for the file name, e.g. the machine")
    parser.add_argument("--note", default="", help="Free text kept in the JSON, e.g. machine and mounting")
    parser.add_argument("--out", type=Path, default=ROOT / "stead_dataset" / "mpu6050")
    parser.add_argument("--force", action="store_true", help="Save even when the checks say the recording is unusable")
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Do not print the stead_dataset coverage summary after saving",
    )
    args = parser.parse_args()

    if args.disturbance and args.state != "STOPPED":
        print("--disturbance is only for STOPPED recordings", file=sys.stderr)
        return 1
    if args.state == "STOPPED" and not args.disturbance:
        args.disturbance = "quiet"

    port = args.port or os.environ.get("SAMS_HW_SERIAL_PORT", "").strip() or "/dev/cu.usbserial-1410"
    try:
        import serial
    except ImportError:
        print("Install pyserial: pip install pyserial", file=sys.stderr)
        return 1

    print(f"Opening {port} @ {args.baud} …")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.5)
    except Exception as exc:
        print(f"FAILED to open port: {exc}", file=sys.stderr)
        print("Stop Django runserver / Serial Monitor, then retry.", file=sys.stderr)
        return 1

    time.sleep(2.0)
    ser.reset_input_buffer()
    booted, chip = wait_for_samples(ser, BOOT_TIMEOUT_S)
    if not booted:
        ser.close()
        print("No samples arrived. Is the mpu6050_serial sketch flashed and the MPU9250/6500 wired? Try: ./hardware/tools/hw.sh flash", file=sys.stderr)
        return 1

    motor = args.motor if args.state == "RUNNING" else "off"
    motor_cmd = MOTOR_OFF if motor == "off" else MOTOR_CMDS[motor]
    if motor_cmd:
        if not send_motor(ser, motor_cmd):
            send_motor(ser, MOTOR_OFF)
            ser.close()
            print(f"The board did not acknowledge {motor_cmd}: reflash it (./hardware/tools/hw.sh flash)", file=sys.stderr)
            return 1
        print(f"Motor: {motor} ({motor_cmd})")
    settle_end = time.monotonic() + SETTLE_S
    while time.monotonic() < settle_end:
        read_line(ser)

    started = datetime.now()
    if args.state == "STOPPED":
        tag = f"_{args.disturbance}" if args.disturbance else ""
    else:
        tag = "_running" if motor == "none" else f"_running_{motor}"
    stem = started.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.name}" if args.name else "") + tag
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{stem}.csv"
    rows: list[tuple[int, float, float, float, float]] = []

    if args.state == "STOPPED":
        hints = {
            "quiet": "keep the machine off and the area still",
            "bumps": "keep the machine off; lightly bump the mount once or twice during the take",
            "footsteps": "keep the machine off; walk near the board during the take",
        }
        doing = hints.get(args.disturbance, "keep the machine off")
    else:
        doing = "keep the machine running as it normally does"
    print(f"Recording {args.seconds:.0f} s {args.state} — {doing}. Ctrl+C stops early.")
    t0 = time.monotonic()
    next_report = 1.0
    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= args.seconds:
                break
            line = read_line(ser)
            if line.startswith("#") and not line.startswith("# TOF raw"):
                print(f"  board: {line[1:].strip()}")
            sample = parse_sample(line)
            if sample is None:
                continue
            rows.append((len(rows), round(elapsed, 4), *sample))
            if elapsed >= next_report:
                next_report += 1.0
                recent = [math.sqrt(x * x + y * y + z * z) for _i, _t, x, y, z in rows[-int(SAMPLE_RATE_HZ):]]
                mean = sum(recent) / len(recent)
                std = math.sqrt(sum((m - mean) ** 2 for m in recent) / len(recent))
                print(f"  {elapsed:5.1f} s  samples={len(rows):5d}  |a| mean={mean:.3f} g  std={std:.4f} g")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        if motor_cmd and motor_cmd != MOTOR_OFF:
            send_motor(ser, MOTOR_OFF)
        ser.close()

    duration = time.monotonic() - t0
    if not rows:
        print("No samples arrived. Is the mpu6050_serial sketch flashed and the MPU9250/6500 wired? Try: ./hardware/tools/hw.sh flash", file=sys.stderr)
        return 1
    problems = recording_problems(rows, duration)
    if problems and not args.force:
        print("Not saved — this recording would teach the model the wrong thing:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "seconds_elapsed", "x", "y", "z"])
        writer.writerows(rows)
    meta = {
        "state": args.state,
        "sensor": f"{chip} (hardware/firmware/mpu6050_serial)",
        "chip": chip,
        "units": "g",
        "firmware_sample_rate_hz": SAMPLE_RATE_HZ,
        "motor": motor,
        "motor_command": motor_cmd or None,
        "disturbance": args.disturbance or None,
        "measured_sample_rate_hz": round(measured_rate(rows, duration), 2),
        "samples": len(rows),
        "duration_s": round(duration, 2),
        "port": port,
        "started_local": started.isoformat(timespec="seconds"),
        "started_utc": started.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "name": args.name,
        "note": args.note,
        "check_problems": problems,
    }
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    windows = max(0, (len(rows) - 100) // 50 + 1)
    print(f"Saved {len(rows)} samples ({meta['measured_sample_rate_hz']} Hz, {windows} training windows) → {csv_path}")
    if not args.skip_coverage:
        try:
            tools_dir = str(Path(__file__).resolve().parent)
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from steady_coverage import coverage_report, print_report

            print()
            print_report(coverage_report(args.out), show_next=True)
        except Exception as exc:
            print(f"(coverage summary skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
