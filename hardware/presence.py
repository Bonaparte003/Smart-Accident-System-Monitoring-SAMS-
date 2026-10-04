"""What the board has actually reported. A missing sensor stays inactive."""

from __future__ import annotations

import threading
import time
from typing import Any


class BoardPresence:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mpu_at = 0.0
        self.tof_present: bool | None = None
        self.tof_at = 0.0
        self.tof_m: float | None = None
        self.buzzer_ack = False

    def reset(self) -> None:
        with self._lock:
            self.mpu_at = 0.0
            self.tof_present = None
            self.tof_at = 0.0
            self.tof_m = None
            self.buzzer_ack = False

    def note_line(self, line: str) -> None:
        text = (line or "").strip()
        if text.startswith("#"):
            text = text[1:].strip()
        if not text:
            return
        with self._lock:
            if text.startswith("TOF ready") or text.startswith("TOF raw="):
                self.tof_present = True
                self.tof_at = time.monotonic()
            elif "TOF not found" in text:
                self.tof_present = False
                self.tof_m = None
            elif text.startswith("ACK B"):
                self.buzzer_ack = True

    def note_sample(self) -> None:
        with self._lock:
            self.mpu_at = time.monotonic()

    def note_tof(self, metres: float) -> None:
        with self._lock:
            self.tof_present = True
            self.tof_at = time.monotonic()
            self.tof_m = float(metres)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                "mpu_age_s": None if not self.mpu_at else round(now - self.mpu_at, 3),
                "tof_present": self.tof_present,
                "tof_age_s": None if not self.tof_at else round(now - self.tof_at, 3),
                "tof_m": None if self.tof_m is None else round(self.tof_m, 3),
                "buzzer_ack": self.buzzer_ack,
            }


_presence: BoardPresence | None = None
_lock = threading.Lock()


def get_presence() -> BoardPresence:
    global _presence
    if _presence is None:
        with _lock:
            if _presence is None:
                _presence = BoardPresence()
    return _presence
