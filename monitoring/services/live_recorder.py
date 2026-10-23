"""Keep every Go Live session as an Archive recording.

Each analysed live frame is written to disk as it arrives, and a log card is
stored every ``CARD_STEP_S`` seconds and whenever the decision changes, so the
session's data survives a restart. When the session ends (Disconnect, a new
QR session, or server start-up finding an unfinished one) the frames become
the clip's video on the real clock. The cards are reviewed and kept in MLOps
like any other recording.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import close_old_connections
from django.utils import timezone

from .recordings import CARD_STEP_S, PROBLEM_LEVELS, card_detail, encode_clip, store_card

logger = logging.getLogger("monitoring.live_recorder")


def frames_dir(recording_pk: int) -> Path:
    return Path(settings.MEDIA_ROOT) / "recordings" / "live_frames" / str(recording_pk)


def _frame_time(path: Path) -> float:
    return int(path.stem) / 1000.0


class LiveRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = None
        self._token: str | None = None
        self._started = 0.0
        self._next_card_t = 0.0
        self._last_level: str | None = None
        self._cards = 0
        self._last_jpeg: bytes | None = None

    @property
    def recording_id(self) -> int | None:
        with self._lock:
            return self._recording.pk if self._recording is not None else None

    def capture(self, token: str | None, jpeg: bytes, payload: dict[str, Any], vision: dict[str, Any] | None) -> None:
        """Add one analysed live frame to the session's recording."""
        if not token or not jpeg:
            return
        with self._lock:
            if token != self._token:
                self._finish_locked()
                self._begin_locked(token)
            if jpeg == self._last_jpeg:
                return
            self._last_jpeg = jpeg
            t = time.monotonic() - self._started
            folder = frames_dir(self._recording.pk)
            (folder / f"{int(t * 1000):09d}.jpg").write_bytes(jpeg)
            level = (payload.get("fusion") or {}).get("label") or ""
            if t >= self._next_card_t or level != self._last_level:
                store_card(self._recording, self._cards, t, card_detail(payload, vision, source="live"), jpeg)
                self._cards += 1
                self._next_card_t = t + CARD_STEP_S
                self._last_level = level

    def finish(self) -> None:
        """End the current session; its video is written in the background."""
        with self._lock:
            self._finish_locked()

    def _begin_locked(self, token: str) -> None:
        from monitoring.models import Recording

        stamp = timezone.localtime().strftime("%Y-%m-%d %H:%M")
        self._recording = Recording.objects.create(title=f"Live {stamp}", source="live", render_status="pending")
        frames_dir(self._recording.pk).mkdir(parents=True, exist_ok=True)
        self._token = token
        self._started = time.monotonic()
        self._next_card_t = 0.0
        self._last_level = None
        self._cards = 0
        self._last_jpeg = None

    def _finish_locked(self) -> None:
        recording = self._recording
        self._recording = None
        self._token = None
        self._last_jpeg = None
        if recording is not None:
            enqueue_finalize(recording.pk)


def enqueue_finalize(recording_pk: int) -> None:
    """Encode the live session on Celery, or a local thread if the broker is down."""
    from monitoring.tasks import finalize_live_recording

    pk = int(recording_pk)
    try:
        finalize_live_recording.delay(pk)
        logger.info("Queued live finalize %s on Celery", pk)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery unavailable for live %s (%s); using a local thread", pk, exc)
    threading.Thread(target=_finalize_in_thread, args=(pk,), name=f"sams-live-{pk}", daemon=True).start()


def finalize(recording_pk: int) -> None:
    """Turn a session's frames into its video. A session without frames is dropped."""
    from monitoring.models import Recording

    recording = Recording.objects.filter(pk=recording_pk).first()
    if recording is None:
        return
    folder = frames_dir(recording_pk)
    frames = sorted(folder.glob("*.jpg")) if folder.exists() else []
    if not frames:
        for mark in recording.minutes.all():
            if mark.snapshot:
                mark.snapshot.delete(save=False)
        recording.delete()
        shutil.rmtree(folder, ignore_errors=True)
        return
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        dest = Path(tmp.name)
    try:
        duration = encode_clip([(_frame_time(path), path) for path in frames], dest)
        recording.video.save(f"live-{recording_pk}.mp4", ContentFile(dest.read_bytes()), save=False)
        recording.duration_s = duration
        recording.fps = round(len(frames) / max(duration, 1e-6), 2)
        recording.problem_count = recording.minutes.filter(level__in=PROBLEM_LEVELS).count()
        recording.render_status = "ready"
        recording.save()
        shutil.rmtree(folder, ignore_errors=True)
    except Exception:
        logger.exception("Could not write the video for live recording %s", recording_pk)
        Recording.objects.filter(pk=recording_pk).update(render_status="failed")
    finally:
        dest.unlink(missing_ok=True)


def _finalize_in_thread(recording_pk: int) -> None:
    close_old_connections()
    try:
        finalize(recording_pk)
    finally:
        close_old_connections()


def finalize_orphans() -> None:
    """Finish live recordings left pending by a restart (none can be running at start-up)."""
    from monitoring.models import Recording

    close_old_connections()
    try:
        for pk in Recording.objects.filter(source="live", render_status="pending").values_list("pk", flat=True):
            enqueue_finalize(pk)
    finally:
        close_old_connections()


_recorder = LiveRecorder()


def get_recorder() -> LiveRecorder:
    return _recorder
