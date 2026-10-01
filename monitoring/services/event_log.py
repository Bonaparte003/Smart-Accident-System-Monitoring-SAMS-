"""In-memory activity log for the SAMS monitoring UI and API."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_events: deque[dict[str, Any]] = deque(maxlen=500)


def log_event(level: str, source: str, message: str, **data: Any) -> dict[str, Any]:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "source": source,
        "message": message,
        "data": data or {},
    }
    with _lock:
        _events.appendleft(event)
    return event


def get_events(limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
    with _lock:
        items = list(_events)
    if source:
        items = [e for e in items if e["source"] == source]
    return items[: max(1, min(limit, 500))]


def clear_events() -> None:
    with _lock:
        _events.clear()
