"""Traffic-light LEDs (green / yellow / red) driven by SAMS risk labels."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Literal

logger = logging.getLogger("hardware.leds")

LedColor = Literal["off", "green", "yellow", "red"]

_LABEL_TO_LED: dict[str, LedColor] = {
    "NORMAL": "green",
    "CAUTION": "yellow",
    "ABNORMAL": "yellow",
    "DANGER": "red",
    "DEGRADED": "yellow",
}

_CMD: dict[LedColor, str] = {
    "off": "L0",
    "green": "LG",
    "yellow": "LY",
    "red": "LR",
}


class Leds:
    """Green / yellow / red status LEDs over the shared MCU serial link."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._color: LedColor = "off"
        self._lock = threading.Lock()
        self._write: Callable[[str], None] | None = None

    def attach_writer(self, write_line: Callable[[str], None] | None) -> None:
        with self._lock:
            self._write = write_line

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "color": self._color,
                "linked": self._write is not None,
            }

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self.enabled = bool(on)
            if not self.enabled:
                self._apply("off")

    def apply_label(self, label: str | None) -> LedColor:
        color = _LABEL_TO_LED.get((label or "NORMAL").upper(), "off")
        self.set_color(color)
        return color

    def set_color(self, color: LedColor) -> None:
        if color not in _CMD:
            raise ValueError(f"Invalid LED color: {color}")
        with self._lock:
            if not self.enabled and color != "off":
                return
            self._apply(color)

    def _apply(self, color: LedColor) -> None:
        prev = self._color
        self._color = color
        cmd = _CMD[color]
        if self._write is not None:
            try:
                self._write(cmd)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LED write failed: %s", exc)
        if prev != color:
            logger.info("LEDs → %s", color)


_leds: Leds | None = None
_leds_lock = threading.Lock()


def get_leds() -> Leds:
    global _leds
    if _leds is None:
        with _leds_lock:
            if _leds is None:
                from .config import load_config

                cfg = load_config()
                _leds = Leds(enabled=cfg.leds_enabled)
    return _leds
