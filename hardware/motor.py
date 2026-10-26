"""D9 motor that simulates a running machine, so the MPU9250/6500 feels real vibration.

Driven by hand from the Live page — never by the risk level, since its vibration
feeds back into the vibration and machine-state models.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Literal

logger = logging.getLogger("hardware.motor")

MotorLevel = Literal["off", "low", "medium", "high", "intermittent"]

_CMD: dict[MotorLevel, str] = {
    "off": "V0",
    "low": "M120",
    "medium": "M180",
    "high": "M255",
    "intermittent": "V1",
}

LEVELS: tuple[MotorLevel, ...] = tuple(_CMD)


class Motor:
    """Machine-simulation motor over the shared MCU serial link."""

    ACK_TIMEOUT_S = 0.6

    def __init__(self) -> None:
        self._level: MotorLevel = "off"
        self._lock = threading.Lock()
        self._write: Callable[[str], None] | None = None
        self._acked = threading.Event()
        self._expect = ""
        self._last_acked: bool | None = None

    def note_ack(self, line: str) -> None:
        """Serial reader hook: the board printed `# ACK …`."""
        if self._expect and line.strip() == f"# ACK {self._expect}":
            self._acked.set()

    def attach_writer(self, write_line: Callable[[str], None] | None) -> None:
        with self._lock:
            self._write = write_line

    def detach(self) -> None:
        """Stop the motor and forget the serial link (stream stopping)."""
        with self._lock:
            if self._write is not None and self._level != "off":
                try:
                    self._write(_CMD["off"])
                except Exception:  # noqa: BLE001
                    pass
            self._write = None
            self._level = "off"

    def status(self) -> dict:
        with self._lock:
            return {
                "level": self._level,
                "running": self._level != "off",
                "linked": self._write is not None,
                "levels": list(LEVELS),
                "acked": self._last_acked,
            }

    def set_level(self, level: str) -> dict:
        level = str(level or "").strip().lower()
        if level not in _CMD:
            raise ValueError(f"Machine level must be one of: {', '.join(LEVELS)}")
        with self._lock:
            if self._write is None:
                raise RuntimeError("Board not connected — the serial stream is not running.")
            self._acked.clear()
            self._expect = _CMD[level]
            self._write(_CMD[level])
            if level != self._level:
                logger.info("Machine motor → %s", level)
            self._level = level  # type: ignore[assignment]
        # Old firmware ignores M<duty> silently; the missing ACK is the only sign of it.
        acked = self._acked.wait(self.ACK_TIMEOUT_S)
        with self._lock:
            self._last_acked = acked
            self._expect = ""
        if not acked:
            logger.warning("Machine motor: no ACK for %s — re-flash the board (hw.sh flash)", _CMD[level])
        return self.status()


_motor: Motor | None = None
_motor_lock = threading.Lock()


def get_motor() -> Motor:
    global _motor
    if _motor is None:
        with _motor_lock:
            if _motor is None:
                _motor = Motor()
    return _motor
