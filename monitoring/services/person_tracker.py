"""Distance history for a person id assigned by the detector's tracker.

Who is who comes from the PPE checkpoint's ByteTrack ids. This only keeps
each id's recent ranges and the least-squares slope: positive ``approach_mps``
means that person is closing on the machine.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

APPROACH_MIN_MPS = 0.15


class MotionLog:
    def __init__(
        self,
        *,
        history_s: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.history_s = history_s
        self._clock = clock
        self._lock = threading.Lock()
        self._distances: dict[int, deque[tuple[float, float]]] = {}

    def reset(self) -> None:
        with self._lock:
            self._distances.clear()

    def record_distance(self, track_id: int, distance_m: float | None, *, now: float | None = None) -> None:
        if distance_m is None:
            return
        now = self._clock() if now is None else now
        with self._lock:
            samples = self._distances.setdefault(int(track_id), deque())
            samples.append((now, float(distance_m)))
            while samples and now - samples[0][0] > self.history_s:
                samples.popleft()

    def motion(self, track_id: int) -> dict[str, Any]:
        """Radial speed and approach rate from this person's recent distances."""
        with self._lock:
            samples = list(self._distances.get(int(track_id), ()))
        still = {"speed_mps": 0.0, "approach_mps": 0.0, "approaching": False, "samples": len(samples)}
        if len(samples) < 3:
            return still
        t = np.array([s[0] for s in samples])
        d = np.array([s[1] for s in samples])
        if t[-1] - t[0] < 0.3:
            return still
        slope = float(np.polyfit(t - t[0], d, 1)[0])
        approach = -slope
        return {
            "speed_mps": round(min(3.0, abs(slope)), 3),
            "approach_mps": round(max(-3.0, min(3.0, approach)), 3),
            "approaching": approach > APPROACH_MIN_MPS,
            "samples": len(samples),
        }

    def prune(self, active_ids: set[int]) -> None:
        with self._lock:
            for track_id in [key for key in self._distances if key not in active_ids]:
                del self._distances[track_id]
