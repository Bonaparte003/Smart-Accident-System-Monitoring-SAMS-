"""Environment-driven hardware settings (real serial MPU6050 only)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HardwareConfig:
    serial_port: str = ""
    serial_baud: int = 115200
    sample_rate_hz: float = 50.0
    window_size: int = 100
    buzzer_auto: bool = True
    buzzer_enabled: bool = True
    # Green is the idle state. Caution and danger replace it.
    leds_auto: bool = True
    leds_enabled: bool = True
    # Android phone over Wi‑Fi (IP Webcam base URL, e.g. http://192.168.1.20:8080)
    camera_url: str = ""
    camera_user: str = ""
    camera_password: str = ""
    camera_timeout_s: float = 2.5


def load_config() -> HardwareConfig:
    return HardwareConfig(
        serial_port=os.environ.get("SAMS_HW_SERIAL_PORT", "").strip(),
        serial_baud=int(os.environ.get("SAMS_HW_SERIAL_BAUD", "115200")),
        sample_rate_hz=float(os.environ.get("SAMS_HW_SAMPLE_RATE", "50")),
        window_size=int(os.environ.get("SAMS_HW_WINDOW_SIZE", "100")),
        buzzer_auto=_env_bool("SAMS_HW_BUZZER_AUTO", True),
        buzzer_enabled=_env_bool("SAMS_HW_BUZZER_ENABLED", True),
        leds_auto=_env_bool("SAMS_HW_LEDS_AUTO", True),
        leds_enabled=_env_bool("SAMS_HW_LEDS_ENABLED", True),
        camera_url=os.environ.get("SAMS_CAM_URL", "").strip(),
        camera_user=os.environ.get("SAMS_CAM_USER", "").strip(),
        camera_password=os.environ.get("SAMS_CAM_PASSWORD", ""),
        camera_timeout_s=float(os.environ.get("SAMS_CAM_TIMEOUT", "2.5")),
    )
