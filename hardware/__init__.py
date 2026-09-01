"""Physical sensor / actuator adapters for SAMS (MPU6050, buzzer, LEDs, Wi‑Fi cam, ToF)."""

from .buzzer import get_buzzer
from .leds import get_leds
from .stream import get_stream
from .tof import get_tof
from .wifi_camera import get_camera

__all__ = ["get_stream", "get_buzzer", "get_leds", "get_camera", "get_tof"]