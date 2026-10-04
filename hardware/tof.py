"""Time-of-flight ranging for near-field worker distance.

Vision covers range beyond 5 m. A ToF reading at or under 5 m replaces that estimate.

Reading order:
1. Explicit override via ``SAMS_HW_TOF_M`` (lab injection / tests)
2. Latest sample from the shared MCU serial stream (``TOF:<metres>`` lines)
3. Unavailable → ``None`` (caller falls back to vision)
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


class ToFSensor:
    """Thread-safe near-range distance reader."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_m: float | None = None
        self._latest_ts: float = 0.0
        self._log_ts: float = 0.0
        self._stale_s = float(os.environ.get("SAMS_HW_TOF_STALE_S", "2.0"))
        self._enabled = os.environ.get("SAMS_HW_TOF_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

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
        if not self._enabled:
            return None
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
            "enabled": self._enabled,
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
