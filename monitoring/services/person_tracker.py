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


class IdKeeper:
    """Person ids for a video analyzed a few frames a second.

    ByteTrack expects every frame; a few frames a second leaves most boxes
    without a confirmed track. Here a box keeps the id of the recent box it
    overlaps most, or, failing that, the nearest recent box within a third of
    its size. Ids not seen for ``max_age_s`` are retired.
    """

    def __init__(self, *, max_age_s: float = 1.5, min_iou: float = 0.2) -> None:
        self.max_age_s = max_age_s
        self.min_iou = min_iou
        self._next = 1
        self._tracks: dict[int, tuple[list[float], float]] = {}

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _near(a: list[float], b: list[float]) -> float:
        """Centre gap in units of the larger box's diagonal."""
        ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        size = max(np.hypot(a[2] - a[0], a[3] - a[1]), np.hypot(b[2] - b[0], b[3] - b[1]), 1.0)
        return float(np.hypot(ax - bx, ay - by) / size)

    def assign(self, boxes: list[list[float]], now: float) -> list[int]:
        for track_id in [k for k, (_box, seen) in self._tracks.items() if now - seen > self.max_age_s]:
            del self._tracks[track_id]
        pairs = []
        for index, box in enumerate(boxes):
            for track_id, (old, _seen) in self._tracks.items():
                iou = self._iou(box, old)
                if iou >= self.min_iou:
                    pairs.append((1.0 + iou, index, track_id))
                else:
                    gap = self._near(box, old)
                    if gap < 0.35:
                        pairs.append((1.0 - gap, index, track_id))
        ids: list[int | None] = [None] * len(boxes)
        taken: set[int] = set()
        for _score, index, track_id in sorted(pairs, reverse=True):
            if ids[index] is None and track_id not in taken:
                ids[index] = track_id
                taken.add(track_id)
        for index, box in enumerate(boxes):
            if ids[index] is None:
                ids[index] = self._next
                self._next += 1
            self._tracks[int(ids[index])] = (list(box), now)
        return [int(i) for i in ids]


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
