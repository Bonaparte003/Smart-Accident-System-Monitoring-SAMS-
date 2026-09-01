"""Physical buzzer driven by SAMS risk labels (shared MCU serial)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Literal

logger = logging.getLogger("hardware.buzzer")

BuzzerState = Literal["off", "caution", "danger"]

_LABEL_TO_STATE: dict[str, BuzzerState] = {
    "NORMAL": "off",
    "CAUTION": "caution",
    "ABNORMAL": "caution",
    "DANGER": "danger",
}

_CMD: dict[BuzzerState, str] = {
    "off": "B0",
    "caution": "B1",
    "danger": "B2",
}

# Keep a short spike audible even if the label flips back to NORMAL quickly.
_MIN_ON_S = {"caution": 1.2, "danger": 1.6}


class Buzzer:
    """Thread-safe buzzer over the shared MCU serial link."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._state: BuzzerState = "off"
        self._lock = threading.Lock()
        self._write: Callable[[str], None] | None = None
        self._hold_until = 0.0
        self._hold_timer: threading.Timer | None = None

    def attach_writer(self, write_line: Callable[[str], None] | None) -> None:
        with self._lock:
            self._write = write_line

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "state": self._state,
                "linked": self._write is not None,
                "hold_until": self._hold_until or None,
            }

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self.enabled = bool(on)
            if not self.enabled:
                self._cancel_hold_locked()
                self._apply("off")

    def apply_label(self, label: str | None) -> BuzzerState:
        state = _LABEL_TO_STATE.get((label or "NORMAL").upper(), "off")
        self.set_state(state)
        return state

    def set_state(self, state: BuzzerState, *, force: bool = False) -> None:
        if state not in _CMD:
            raise ValueError(f"Invalid buzzer state: {state}")
        with self._lock:
            if not self.enabled and state != "off":
                return
            now = time.monotonic()
            if state == "off":
                if not force and now < self._hold_until:
                    return
                self._cancel_hold_locked()
                self._apply("off")
                return

            self._apply(state)
            hold = _MIN_ON_S.get(state, 0.0)
            if hold > 0:
                self._hold_until = now + hold
                self._cancel_hold_locked()
                timer = threading.Timer(hold, self._hold_expired)
                timer.daemon = True
                self._hold_timer = timer
                timer.start()

    def pulse_test(self, seconds: float = 1.0) -> dict:
        """Force an audible chirp for hardware bring-up (D8 buzzer)."""
        with self._lock:
            if self._write is None:
                raise RuntimeError(
                    "Buzzer not linked — switch to Live first so the serial port is open."
                )
            self.enabled = True
            self._cancel_hold_locked()
            # Repeat so the MCU is more likely to catch B1 while streaming accel.
            for _ in range(3):
                self._write("B1")
            self._state = "caution"
            self._hold_until = time.monotonic() + float(seconds)

        def _stop() -> None:
            self.set_state("off", force=True)

        timer = threading.Timer(float(seconds), _stop)
        timer.daemon = True
        with self._lock:
            self._hold_timer = timer
        timer.start()
        logger.info("Buzzer TEST pulse %.1fs", seconds)
        return self.status()

    def _hold_expired(self) -> None:
        with self._lock:
            self._hold_timer = None
            self._hold_until = 0.0
            # Leave current pattern until next label update; do not force off.

    def _cancel_hold_locked(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None
        self._hold_until = 0.0

    def _apply(self, state: BuzzerState) -> None:
        prev = self._state
        self._state = state
        # Avoid flooding the MCU with repeated B0/B1 (was drowning commands).
        if prev == state and state == "off":
            return
        cmd = _CMD[state]
        if self._write is not None:
            try:
                self._write(cmd)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Buzzer write failed: %s", exc)
        if prev != state:
            logger.info("Buzzer → %s", state)


_buzzer: Buzzer | None = None
_buzzer_lock = threading.Lock()


def get_buzzer() -> Buzzer:
    global _buzzer
    if _buzzer is None:
        with _buzzer_lock:
            if _buzzer is None:
                from .config import load_config

                cfg = load_config()
                _buzzer = Buzzer(enabled=cfg.buzzer_enabled)
    return _buzzer
