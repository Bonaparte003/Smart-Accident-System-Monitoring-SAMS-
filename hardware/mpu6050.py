"""MPU6050 accelerometer over USB serial (CSV lines).

All serial I/O is serialized on one lock so actuator commands (buzzer/LEDs)
are not dropped while the reader thread is mid-readline.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterator

logger = logging.getLogger("hardware.mpu6050")


class AccelSource(ABC):
    """Yields (ax, ay, az) samples in g."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read_sample(self) -> tuple[float, float, float]: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    def write_line(self, line: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support writes")

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SerialMPU6050(AccelSource):
    """Read CSV accel lines and write actuator commands on one UART."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._lock = threading.RLock()
        self._pending: deque[str] = deque()

    def open(self) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for live MPU6050. pip install pyserial"
            ) from exc
        if not self.port:
            raise RuntimeError(
                "SAMS_HW_SERIAL_PORT is empty. Set it to your Arduino/USB serial device "
                "(e.g. /dev/cu.usbserial-1410 or COM3)."
            )
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # CH340/FTDI boards often reset on open — wait for sketch boot.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        logger.info("SerialMPU6050 open on %s @ %s", self.port, self.baud)

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:  # noqa: BLE001
                    pass
                self._ser = None
            self._pending.clear()

    @property
    def connected(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    def write_line(self, line: str) -> None:
        """Queue a command and flush it under the serial lock."""
        payload = line.strip()
        if not payload:
            return
        with self._lock:
            self._pending.append(payload)
            self._flush_commands_locked()

    def _flush_commands_locked(self) -> None:
        ser = self._ser
        if ser is None:
            return
        while self._pending:
            cmd = self._pending.popleft()
            raw = (cmd if cmd.endswith("\n") else f"{cmd}\n").encode("ascii", errors="ignore")
            try:
                ser.write(raw)
                ser.flush()
                logger.info("MCU TX %s", cmd)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MCU TX failed (%s): %s", cmd, exc)
                self._pending.appendleft(cmd)
                break

    def read_sample(self) -> tuple[float, float, float]:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                ser = self._ser
                if ser is None:
                    raise RuntimeError("SerialMPU6050 not open")
                self._flush_commands_locked()
                raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                logger.info("MCU RX %s", line)
                continue
            # Optional ToF firmware lines: TOF:1.23
            if line.upper().startswith("TOF:"):
                try:
                    from .tof import get_tof

                    get_tof().ingest_line(line)
                except Exception:  # noqa: BLE001
                    pass
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
        raise TimeoutError(f"No valid MPU6050 sample on {self.port}")


def iter_samples(source: AccelSource) -> Iterator[tuple[float, float, float]]:
    while True:
        yield source.read_sample()
