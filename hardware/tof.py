"""
Time-of-flight ranging for near-field worker distance
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

logger = logging.getLogger("hardware.tof")

_TOF_LINE = re.compile(r"^TOF:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
OFFLINE_AFTER_S = 5.0


class ToFSensor:
    """Thread-safe near-range distance reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_m: float | None = None
        self._latest_ts: float = 0.0
        self._log_ts: float = 0.0
        self._stale_s = float(os.environ.get("SAMS_HW_TOF_STALE_S", "2.0"))

    def online(self) -> bool:
        """The board reported the sensor recently, or a lab override is set."""
        if os.environ.get("SAMS_HW_TOF_M", "").strip():
            return True
        from .presence import get_presence

        board = get_presence().snapshot()
        age = board.get("tof_age_s")
        return board.get("tof_present") is True and age is not None and age <= OFFLINE_AFTER_S

    def ingest_line(self, line: str) -> bool:
        """Parse a firmware line like ``TOF:1.23``. Returns True if consumed."""
        m = _TOF_LINE.match((line or "").strip())
        if not m:
            return False
        try:
            metres = float(m.group(1))
        except ValueError:
            return False
        if metres <= 0.01 or metres > 20.0:
            return False
        now = time.monotonic()
        with self._lock:
            self._latest_m = metres
            self._latest_ts = now
            should_log = now - self._log_ts >= 1.0
            if should_log:
                self._log_ts = now
        from .presence import get_presence

        get_presence().note_tof(metres)
        if should_log:
            logger.info("ToF %.2f m", metres)
        return True

    def push_metres(self, metres: float) -> None:
        """Manual inject (tests / env override path)."""
        with self._lock:
            self._latest_m = float(metres)
            self._latest_ts = time.monotonic()

    def read_metres(self) -> float | None:
        override = os.environ.get("SAMS_HW_TOF_M", "").strip()
        if override:
            try:
                val = float(override)
                if val > 0.01:
                    return val
            except ValueError:
                logger.warning("Invalid SAMS_HW_TOF_M=%r", override)

        with self._lock:
            if self._latest_m is None:
                return None
            age = time.monotonic() - self._latest_ts
            if age > self._stale_s:
                return None
            return float(self._latest_m)

    def status(self) -> dict[str, Any]:
        metres = self.read_metres()
        with self._lock:
            age = (time.monotonic() - self._latest_ts) if self._latest_ts else None
        return {
            "online": self.online(),
            "available": metres is not None,
            "distance_m": round(metres, 3) if metres is not None else None,
            "age_s": round(age, 3) if age is not None else None,
            "stale_s": self._stale_s,
            "override_env": bool(os.environ.get("SAMS_HW_TOF_M", "").strip()),
        }


_tof: ToFSensor | None = None
_tof_lock = threading.Lock()


def get_tof() -> ToFSensor:
    global _tof
    if _tof is None:
        with _tof_lock:
            if _tof is None:
                _tof = ToFSensor()
    return _tof
