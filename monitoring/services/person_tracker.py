"""Distance history for a person id assigned by the detector's tracker.

Who is who comes from the PPE checkpoint's ByteTrack ids. This only keeps
each id's ranges over the last ``history_s`` seconds (the window the fusion
model's motion inputs were measured over) and their least-squares slope: positive
``approach_mps`` means that person is closing on the machine. With fewer than
two ranges there is no motion to report.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np


class MotionLog:
    def __init__(
        self,
        *,
        history_s: float,
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
        t = np.array([s[0] for s in samples])
        if len(samples) < 2 or t[-1] == t[0]:
            return {"speed_mps": None, "approach_mps": None, "approaching": None, "samples": len(samples)}
        d = np.array([s[1] for s in samples])
        slope = float(np.polyfit(t - t[0], d, 1)[0])
        return {
            "speed_mps": round(abs(slope), 3),
            "approach_mps": round(-slope, 3),
            "approaching": -slope > 0,
            "samples": len(samples),
        }

    def prune(self, active_ids: set[int]) -> None:
        with self._lock:
            for track_id in [key for key in self._distances if key not in active_ids]:
                del self._distances[track_id]
