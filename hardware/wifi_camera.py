"""Phone camera via QR join — phone scans Mac QR and pushes JPEG frames over Wi‑Fi."""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import socket
import threading
import time
from typing import Any

from PIL import Image

logger = logging.getLogger("hardware.camera")


def guess_lan_ip() -> str:
    """Best-effort LAN IPv4 for QR join URLs (phone must reach this host)."""
    override = os.environ.get("SAMS_PUBLIC_HOST", "").strip()
    if override:
        # Allow host or host:port — caller adds scheme/port as needed
        return override.split("://")[-1].split("/")[0].split(":")[0]
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.4)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


class PhoneCamera:
    """Desktop waits with a QR; phone opens join URL and uploads frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._live = False
        self._phone_joined = False
        self._last_error: str | None = None
        self._last_grab_at: float | None = None
        self._last_jpeg: bytes | None = None
        self._last_prediction: dict[str, Any] | None = None
        self._grab_count = 0
        self._join_base: str = ""

    @property
    def token(self) -> str | None:
        with self._lock:
            return self._token

    @property
    def is_live(self) -> bool:
        with self._lock:
            return self._live and bool(self._token)

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._last_jpeg is not None

    def set_join_base(self, base_url: str) -> None:
        with self._lock:
            self._join_base = (base_url or "").rstrip("/")

    def start_session(self, *, join_base: str | None = None) -> dict[str, Any]:
        """Create a fresh token; previous session is invalidated."""
        token = secrets.token_urlsafe(12)
        with self._lock:
            if join_base:
                self._join_base = join_base.rstrip("/")
            self._token = token
            self._live = True
            self._phone_joined = False
            self._last_error = None
            self._last_jpeg = None
            self._last_grab_at = None
            self._last_prediction = None
            self._grab_count = 0
        return self.status()

    def stop_session(self) -> dict[str, Any]:
        with self._lock:
            self._live = False
            self._token = None
            self._phone_joined = False
            self._last_error = None
        return self.status()

    def set_live(self, live: bool) -> None:
        if live:
            if not self._token:
                self.start_session()
            else:
                with self._lock:
                    self._live = True
        else:
            self.stop_session()

    def validate_token(self, token: str) -> bool:
        with self._lock:
            return bool(self._live and self._token and secrets.compare_digest(self._token, token or ""))

    def mark_joined(self, token: str) -> None:
        if not self.validate_token(token):
            raise PermissionError("Invalid or expired camera session")
        with self._lock:
            self._phone_joined = True
            self._last_error = None

    def push_jpeg(self, token: str, data: bytes) -> dict[str, Any]:
        if not self.validate_token(token):
            raise PermissionError("Invalid or expired camera session")
        if not data or len(data) < 100:
            raise ValueError("Empty frame")
        Image.open(io.BytesIO(data)).verify()
        with self._lock:
            self._phone_joined = True
            self._last_jpeg = data
            self._last_grab_at = time.time()
            self._grab_count += 1
            self._last_error = None
        return self.status()

    def grab_jpeg(self) -> bytes:
        """The phone's latest JPEG, exactly as it was uploaded."""
        with self._lock:
            raw = self._last_jpeg
            age = None if self._last_grab_at is None else time.time() - self._last_grab_at
        if not raw:
            raise RuntimeError("Waiting for phone — scan the QR and allow camera access")
        if age is not None and age > 8.0:
            raise RuntimeError("Phone feed stalled — keep the join page open on the phone")
        return raw

    def grab_image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.grab_jpeg())).convert("RGB")

    def record_prediction(self, prediction: dict[str, Any] | None) -> None:
        with self._lock:
            self._last_prediction = prediction

    def last_prediction(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_prediction) if self._last_prediction else None

    def last_thumbnail_data_url(self, max_side: int = 480, quality: int = 75) -> str | None:
        with self._lock:
            raw = self._last_jpeg
        if not raw:
            return None
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((max_side, max_side), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def join_url(self) -> str | None:
        with self._lock:
            return self._join_url_unlocked()

    def _join_url_unlocked(self) -> str | None:
        if not self._token or not self._join_base:
            return None
        return f"{self._join_base}/cam/{self._token}/"

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = None
            if self._last_grab_at is not None:
                age = round(time.time() - self._last_grab_at, 2)
            join = self._join_url_unlocked()
            return {
                "mode": "qr_join",
                "live": self._live,
                "configured": bool(self._token),
                "token": self._token,
                "join_url": join,
                "join_base": self._join_base,
                "phone_joined": self._phone_joined,
                "grab_count": self._grab_count,
                "last_grab_age_s": age,
                "last_error": self._last_error,
                "has_frame": self._last_jpeg is not None,
                "last_label": (self._last_prediction or {}).get("label"),
                "last_unsafe": (self._last_prediction or {}).get("unsafe"),
                "url": join or "",
                "host": "phone" if self._phone_joined else ("qr" if self._token else ""),
                "auth": False,
            }


_camera: PhoneCamera | None = None
_camera_lock = threading.Lock()


def get_camera() -> PhoneCamera:
    global _camera
    if _camera is None:
        with _camera_lock:
            if _camera is None:
                _camera = PhoneCamera()
    return _camera
