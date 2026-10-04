"""Background MPU6050 window buffer for live inference (serial only)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Literal

import numpy as np

from .buzzer import get_buzzer
from .config import HardwareConfig, load_config
from .leds import get_leds
from .mpu6050 import AccelSource, SerialMPU6050
from .presence import get_presence

logger = logging.getLogger("hardware.stream")

Mode = Literal["simulate", "live"]


def apply_risk_actuators(label: str | None, *, buzzer: bool = True, leds: bool = True) -> None:
    """Drive buzzer + LEDs from a fusion / vibration label."""
    if buzzer:
        get_buzzer().apply_label(label)
    if leds:
        get_leds().apply_label(label)


class AccelStream:
    """Collects real MPU6050 samples into fixed-size windows for the vibration model."""

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or load_config()
        self.mode: Mode = "simulate"
        self._source: AccelSource | None = None
        self._buf: deque[tuple[float, float, float]] = deque(maxlen=self.config.window_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._samples_total = 0
        self._started_at: float | None = None
        self._last_sample_at: float | None = None
        self._last_prediction: dict[str, Any] | None = None
        self._buzzer_auto = self.config.buzzer_auto
        self._leds_auto = self.config.leds_auto
        self._machine_on = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            filled = len(self._buf)
            return {
                "mode": self.mode,
                "source": "serial",
                "connected": bool(self._source and self._source.connected),
                "running": self._thread is not None and self._thread.is_alive(),
                "window_size": self.config.window_size,
                "sample_rate_hz": self.config.sample_rate_hz,
                "buffer_filled": filled,
                "window_ready": filled >= self.config.window_size,
                "samples_total": self._samples_total,
                "last_error": self._last_error,
                "started_at": self._started_at,
                "last_sample_at": self._last_sample_at,
                "buzzer_auto": self._buzzer_auto,
                "leds_auto": self._leds_auto,
                "machine_on": self._machine_on,
                "last_label": (self._last_prediction or {}).get("label"),
                "serial_port": self.config.serial_port or None,
            }

    def set_mode(self, mode: Mode, *, source: str | None = None) -> dict[str, Any]:
        if mode not in ("simulate", "live"):
            raise ValueError("mode must be 'simulate' or 'live'")
        if source and source.strip().lower() not in ("", "serial"):
            raise ValueError("Only serial MPU6050 is supported (no mock source)")
        if mode == "live":
            self.start()
        else:
            self.stop()
            get_buzzer().set_state("off")
            get_leds().set_color("off")
        self.mode = mode
        logger.info("Hardware mode → %s", self.mode)
        return self.status()

    def set_buzzer_auto(self, on: bool) -> None:
        self._buzzer_auto = bool(on)

    def set_leds_auto(self, on: bool) -> None:
        self._leds_auto = bool(on)

    def set_machine_on(self, on: bool) -> None:
        """Explicit run switch. A vibration sample does not turn the machine on."""
        self._machine_on = bool(on)

    @property
    def machine_on(self) -> bool:
        return self._machine_on

    @property
    def buzzer_auto(self) -> bool:
        return self._buzzer_auto

    @property
    def leds_auto(self) -> bool:
        return self._leds_auto

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stop()
        get_presence().reset()
        self._stop.clear()
        self._source = SerialMPU6050(
            port=self.config.serial_port,
            baud=self.config.serial_baud,
        )
        try:
            self._source.open()
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self._source = None
            get_buzzer().attach_writer(None)
            get_leds().attach_writer(None)
            logger.warning("Failed to open MPU6050 serial: %s", exc)
            raise
        get_buzzer().attach_writer(self._source.write_line)
        get_leds().attach_writer(self._source.write_line)
        if get_leds().enabled:
            get_leds().set_color("green")
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="sams-mpu6050", daemon=True)
        self._thread.start()
        # Raw serial chirp — proves D8 wiring without waiting for a risk label.
        source = self._source

        def _chirp_off() -> None:
            try:
                if source is not None and source.connected:
                    source.write_line("B0")
            except Exception:  # noqa: BLE001
                pass

        try:
            source.write_line("B1")
            threading.Timer(0.45, _chirp_off).start()
            logger.info("Buzzer link chirp sent (B1)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Buzzer link chirp failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        get_buzzer().attach_writer(None)
        get_leds().attach_writer(None)
        if self._source is not None:
            try:
                self._source.close()
            except Exception:  # noqa: BLE001
                pass
            self._source = None
        get_presence().reset()
        with self._lock:
            self._buf.clear()

    def _run(self) -> None:
        assert self._source is not None
        while not self._stop.is_set():
            try:
                sample = self._source.read_sample()
                with self._lock:
                    self._buf.append(sample)
                    self._samples_total += 1
                    self._last_sample_at = time.time()
                    self._last_error = None
                get_presence().note_sample()
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                logger.warning("MPU6050 read error: %s", exc)
                time.sleep(0.2)

    def latest_window(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buf) < self.config.window_size:
                return None
            arr = np.array(self._buf, dtype=np.float32)
        return arr

    def record_prediction(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._last_prediction = result

    def last_prediction(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_prediction) if self._last_prediction else None

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


_stream: AccelStream | None = None
_stream_lock = threading.Lock()


def get_stream() -> AccelStream:
    global _stream
    if _stream is None:
        with _stream_lock:
            if _stream is None:
                _stream = AccelStream()
    return _stream
