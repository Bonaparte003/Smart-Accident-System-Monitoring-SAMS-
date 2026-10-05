"""One background worker that builds log cards and the framed video."""

from __future__ import annotations

import logging
import queue
import threading

from django.db import close_old_connections

logger = logging.getLogger("monitoring.video_jobs")

_queue: queue.Queue[int] = queue.Queue()
_lock = threading.Lock()
_thread: threading.Thread | None = None
_queued: set[int] = set()


def start_worker() -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_run, name="sams-video-jobs", daemon=True)
        _thread.start()


def enqueue_recording(pk: int) -> None:
    start_worker()
    _put(int(pk))


def resume_pending() -> None:
    """Pick up clips that do not yet have a framed copy.

    Called from server startup, which runs on the async loop. The database
    read happens on a thread so Django allows it.
    """
    start_worker()
    threading.Thread(target=_enqueue_pending, name="sams-video-resume", daemon=True).start()


def _enqueue_pending() -> None:
    close_old_connections()
    try:
        from monitoring.models import Recording

        pending = set(Recording.objects.filter(render_status="pending").values_list("pk", flat=True))
        pending.update(Recording.objects.filter(framed="").values_list("pk", flat=True))
        for pk in sorted(pending):
            _put(int(pk))
        print(f"Video jobs queued: {len(pending)}", flush=True)
    finally:
        close_old_connections()


def _put(pk: int) -> None:
    with _lock:
        if pk in _queued:
            return
        _queued.add(pk)
    _queue.put(pk)


def _run() -> None:
    while True:
        pk = _queue.get()
        close_old_connections()
        print(f"Video job {pk} started", flush=True)
        try:
            from monitoring.services.recordings import finish_recording

            finish_recording(pk)
            print(f"Video job {pk} finished", flush=True)
        except Exception:
            logger.exception("Video job %s failed", pk)
            print(f"Video job {pk} failed", flush=True)
            try:
                from monitoring.models import Recording

                Recording.objects.filter(pk=pk).update(render_status="failed")
            except Exception:
                logger.exception("Could not mark video job %s failed", pk)
        finally:
            with _lock:
                _queued.discard(pk)
            close_old_connections()
            _queue.task_done()
