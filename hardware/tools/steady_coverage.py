#!/usr/bin/env python3
"""
Report what machine-state recordings are still missing under stead_dataset/mpu6050/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "stead_dataset" / "mpu6050"

MIN_STOPPED = 3
MIN_PER_MOTOR = 2
RUNNING_MOTORS = ("low", "medium", "high", "intermittent")
STOPPED_DISTURBANCES = ("quiet", "bumps", "footsteps")


def load_recordings(folder: Path) -> list[dict]:
    rows: list[dict] = []
    if not folder.is_dir():
        return rows
    for meta_path in sorted(folder.glob("*.json")):
        csv_path = meta_path.with_suffix(".csv")
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            rows.append({"stem": meta_path.stem, "error": str(exc), "path": str(meta_path)})
            continue
        state = str(meta.get("state") or "").upper()
        motor = str(meta.get("motor") or ("off" if state == "STOPPED" else "unknown")).lower()
        note = str(meta.get("note") or "").lower()
        disturbance = str(meta.get("disturbance") or "").lower()
        if not disturbance:
            for label in STOPPED_DISTURBANCES:
                if label in note or label in meta_path.stem.lower():
                    disturbance = label
                    break
            if state == "STOPPED" and not disturbance:
                disturbance = "quiet"
        rows.append(
            {
                "stem": meta_path.stem,
                "state": state,
                "motor": motor,
                "disturbance": disturbance,
                "note": meta.get("note") or "",
                "name": meta.get("name") or "",
                "samples": meta.get("samples"),
                "duration_s": meta.get("duration_s"),
                "chip": meta.get("chip"),
                "has_csv": csv_path.is_file(),
                "path": str(meta_path),
            }
        )
    return rows


def coverage_report(folder: Path) -> dict:
    recordings = load_recordings(folder)
    stopped = [row for row in recordings if row.get("state") == "STOPPED" and row.get("has_csv")]
    running = [row for row in recordings if row.get("state") == "RUNNING" and row.get("has_csv")]
    by_motor: dict[str, list] = defaultdict(list)
    for row in running:
        by_motor[str(row.get("motor") or "unknown")].append(row)
    by_disturbance: dict[str, list] = defaultdict(list)
    for row in stopped:
        by_disturbance[str(row.get("disturbance") or "quiet")].append(row)

    gaps: list[str] = []
    next_cmds: list[str] = []
    name = next((row.get("name") for row in recordings if row.get("name")), "rig01")

    if len(stopped) < MIN_STOPPED:
        gaps.append(f"STOPPED: have {len(stopped)}, need at least {MIN_STOPPED} (quiet, bumps, footsteps)")
    for label in STOPPED_DISTURBANCES:
        if len(by_disturbance.get(label, [])) < 1:
            gaps.append(f"STOPPED/{label}: none yet")
            note = {
                "quiet": "machine off, quiet",
                "bumps": "machine off, bump the mount once or twice",
                "footsteps": "machine off, walk near the board",
            }[label]
            next_cmds.append(
                f'./hardware/tools/hw.sh record-steady --seconds 60 --name {name} '
                f'--disturbance {label} --note "{note}"'
            )

    for motor in RUNNING_MOTORS:
        have = len(by_motor.get(motor, []))
        if have < MIN_PER_MOTOR:
            gaps.append(f"RUNNING/{motor}: have {have}, need at least {MIN_PER_MOTOR}")
            for take in range(have + 1, MIN_PER_MOTOR + 1):
                next_cmds.append(
                    f"./hardware/tools/hw.sh record-running --seconds 60 --name {name} "
                    f"--motor {motor} --note \"take {take} of {MIN_PER_MOTOR}\""
                )

    chips = sorted({row.get("chip") for row in recordings if row.get("chip")})
    if len(chips) > 1:
        gaps.append(f"mixed chips {chips}: the model may learn the sensor, not the machine state")

    ready = not gaps
    return {
        "folder": str(folder),
        "recordings": recordings,
        "stopped": stopped,
        "running": running,
        "by_motor": {key: len(value) for key, value in sorted(by_motor.items())},
        "by_disturbance": {key: len(value) for key, value in sorted(by_disturbance.items())},
        "gaps": gaps,
        "next_cmds": next_cmds,
        "ready": ready,
        "targets": {
            "min_stopped": MIN_STOPPED,
            "min_per_motor": MIN_PER_MOTOR,
            "running_motors": list(RUNNING_MOTORS),
            "stopped_disturbances": list(STOPPED_DISTURBANCES),
        },
    }


def print_report(report: dict, *, show_next: bool) -> None:
    print(f"Machine-state coverage under {report['folder']}")
    print()
    print(f"STOPPED  {len(report['stopped'])}  (target ≥ {report['targets']['min_stopped']})")
    for label in report["targets"]["stopped_disturbances"]:
        count = report["by_disturbance"].get(label, 0)
        mark = "ok" if count >= 1 else "MISSING"
        print(f"  {label:12s} {count:3d}  [{mark}]")
    print()
    print(f"RUNNING  {len(report['running'])}  (target ≥ {report['targets']['min_per_motor']} per motor)")
    for motor in report["targets"]["running_motors"]:
        count = report["by_motor"].get(motor, 0)
        mark = "ok" if count >= report["targets"]["min_per_motor"] else "MISSING"
        print(f"  {motor:12s} {count:3d}  [{mark}]")
    extras = [motor for motor in report["by_motor"] if motor not in report["targets"]["running_motors"]]
    for motor in extras:
        print(f"  {motor:12s} {report['by_motor'][motor]:3d}  [extra]")
    print()
    if report["ready"]:
        print("Coverage is enough to re-run notebook 02 §8. Check leave-one-recording-out before promoting.")
    else:
        print("Gaps:")
        for gap in report["gaps"]:
            print(f"  - {gap}")
    if show_next and report["next_cmds"]:
        print()
        print("Next commands (stop Django first if it holds the serial port):")
        for cmd in report["next_cmds"][:8]:
            print(f"  {cmd}")
        if len(report["next_cmds"]) > 8:
            print(f"  … and {len(report['next_cmds']) - 8} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show stead_dataset coverage for the machine-state model")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--next", action="store_true", help="Print the next recording commands to close gaps")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args()
    report = coverage_report(args.dir)
    if args.json:
        payload = dict(report)
        payload["recordings"] = [
            {key: row.get(key) for key in ("stem", "state", "motor", "disturbance", "note", "samples", "has_csv")}
            for row in report["recordings"]
        ]
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print_report(report, show_next=args.next or not report["ready"])
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
