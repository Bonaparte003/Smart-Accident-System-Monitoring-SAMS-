"""Background MPU9250/6500 window buffer for live inference (serial only)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from .buzzer import get_buzzer
from .config import HardwareConfig, load_config
from .leds import get_leds
from .motor import get_motor
from .mpu6050 import AccelSource, SerialMPU6050
from .presence import get_presence

logger = logging.getLogger("hardware.stream")


def apply_risk_actuators(label: str | None, *, buzzer: bool = True, leds: bool = True) -> None:
    """Drive buzzer + LEDs from a fusion / vibration label."""
    if buzzer:
        get_buzzer().apply_label(label)
    if leds:
        get_leds().apply_label(label)


class AccelStream:
    """Collects real MPU9250/6500 samples into fixed-size windows for the vibration model."""

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or load_config()
        self._live = False
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

    def status(self) -> dict[str, Any]:
        with self._lock:
            filled = len(self._buf)
            return {
                "live": self._live,
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
                "buzzer_auto": self.buzzer_auto,
                "leds_auto": self.leds_auto,
                "last_label": (self._last_prediction or {}).get("label"),
                "serial_port": self.config.serial_port or None,
            }

    @property
    def buzzer_auto(self) -> bool:
        return self.config.buzzer_auto

    @property
    def leds_auto(self) -> bool:
        return self.config.leds_auto

    def start(self) -> None:
        """Open the serial board and stream samples. Raises if the port does not open."""
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
            get_motor().detach()
            logger.warning("Failed to open MPU9250/6500 serial: %s", exc)
            raise
        get_buzzer().attach_writer(self._source.write_line)
        get_leds().attach_writer(self._source.write_line)
        get_motor().attach_writer(self._source.write_line)
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="sams-mpu6050", daemon=True)
        self._thread.start()
        self._live = True
        logger.info("Hardware live on %s", self.config.serial_port)
        # Startup sequence — buzzer, LEDs one by one, motor — proves every actuator
        # without waiting for a risk label. The board skips it if it just booted.
        try:
            self._source.write_line("ST")
            logger.info("Startup sequence sent (ST)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Startup sequence failed: %s", exc)

    def stop(self) -> None:
        self._live = False
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        get_buzzer().attach_writer(None)
        get_leds().attach_writer(None)
        get_motor().detach()
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
                logger.warning("MPU9250/6500 read error: %s", exc)
                time.sleep(0.2)

    def latest_window(self, size: int | None = None) -> np.ndarray | None:
        """The newest ``size`` samples (the buffer length when omitted), or None until that many arrived.

        A model that needs a longer window than the buffer holds grows the buffer.
        """
        size = int(size or self.config.window_size)
        with self._lock:
            if size > (self._buf.maxlen or 0):
                self._buf = deque(self._buf, maxlen=size)
            if len(self._buf) < size:
                return None
            arr = np.array(list(self._buf)[-size:], dtype=np.float32)
        return arr

    def record_prediction(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._last_prediction = result

    def last_prediction(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_prediction) if self._last_prediction else None

    @property
    def is_live(self) -> bool:
        return self._live


_stream: AccelStream | None = None
_stream_lock = threading.Lock()


def get_stream() -> AccelStream:
    global _stream
    if _stream is None:
        with _stream_lock:
            if _stream is None:
                _stream = AccelStream()
    return _stream
