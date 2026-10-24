"""Celery tasks for long-running recording work."""

from __future__ import annotations

import logging

from celery import shared_task
from django.db import close_old_connections

logger = logging.getLogger("monitoring.tasks")


@shared_task(name="monitoring.analyze_upload_recording", acks_late=True, reject_on_worker_lost=True)
def analyze_upload_recording(recording_pk: int) -> str:
    """Analyse one uploaded clip with the same models as Live."""
    from monitoring.models import Recording
    from monitoring.services.video_analysis import analyze_recording

    close_old_connections()
    try:
        analyze_recording(int(recording_pk))
        return f"ready:{recording_pk}"
    except Exception:
        logger.exception("Video analysis %s failed", recording_pk)
        Recording.objects.filter(pk=recording_pk).update(render_status="failed")
        raise
    finally:
        close_old_connections()


@shared_task(name="monitoring.finalize_live_recording", acks_late=True, reject_on_worker_lost=True)
def finalize_live_recording(recording_pk: int) -> str:
    """Encode a finished Go Live session into its archive video."""
    from monitoring.services.live_recorder import finalize

    close_old_connections()
    try:
        finalize(int(recording_pk))
        return f"finalized:{recording_pk}"
    finally:
        close_old_connections()
