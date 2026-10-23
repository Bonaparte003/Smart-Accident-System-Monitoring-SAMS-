"""Analyse an uploaded clip with the same models and fusion as Live.

A file carries pixels only, so the sensor inputs fusion needs come from what
the operator declared at upload: the machine state (STOPPED / RUNNING) and the
vibration state (NORMAL / ABNORMAL). A declared input is marked ``declared``
on every card, and one left undeclared stays unmeasured, so those frames get
no decision, as on Live. People are tracked through the clip with the clip's
own tracker and clock, so motion comes from the video itself.

Queued work runs on Celery (``monitoring.tasks.analyze_upload_recording``).
If the broker is down, the same job falls back to a local daemon thread.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from django.db import close_old_connections

from . import machine_state_inference, vibration_inference
from .recordings import CARD_STEP_S, PROBLEM_LEVELS, card_detail, store_card

logger = logging.getLogger("monitoring.video_analysis")

ANALYSIS_FPS = float(os.environ.get("SAMS_VIDEO_ANALYSIS_FPS", "2"))


def _states(module) -> list[str]:
    """The classes a board model outputs, from its metadata; none without the model."""
    return list(module.get_metadata().get("class_names") or [])


def clean_declared(machine_state: str | None, vibration: str | None) -> dict[str, str]:
    """The operator's statements, keeping only states the board models can output."""
    declared = {}
    if (machine_state or "").upper() in _states(machine_state_inference):
        declared["machine_state"] = machine_state.upper()
    if (vibration or "").upper() in _states(vibration_inference):
        declared["vibration"] = vibration.upper()
    return declared


def declared_inputs(declared: dict[str, Any]) -> tuple[dict | None, dict | None]:
    """Vibration and machine-state results standing in for the absent board.

    A declared vibration state has no model probability behind it, so its
    anomaly score is the stated state itself: 1.0 for ABNORMAL, 0.0 for NORMAL.
    """
    vibration = None
    if declared.get("vibration") in _states(vibration_inference):
        abnormal = declared["vibration"] == "ABNORMAL"
        vibration = {
            "label": declared["vibration"],
            "abnormal": abnormal,
            "anomaly_score": 1.0 if abnormal else 0.0,
            "confidence": None,
            "declared": True,
        }
    machine = None
    if declared.get("machine_state") in _states(machine_state_inference):
        machine = {
            "label": declared["machine_state"],
            "running": declared["machine_state"] == "RUNNING",
            "confidence": None,
            "model_id": "declared",
            "declared": True,
        }
    return vibration, machine


def analyze_recording(pk: int) -> None:
    """Write the cards for one uploaded clip and mark it ready."""
    import cv2
    from PIL import Image

    from monitoring.models import Recording

    from .control_center import assemble
    from .vision_inference import VideoTracking, predict_image

    recording = Recording.objects.get(pk=pk)
    for mark in list(recording.minutes.all()):
        if mark.snapshot:
            mark.snapshot.delete(save=False)
        mark.delete()
    vibration, machine = declared_inputs(recording.declared or {})
    cap = cv2.VideoCapture(recording.video.path)
    if not cap.isOpened():
        raise ValueError("Could not open the video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    tracking = VideoTracking()
    step = 1.0 / max(ANALYSIS_FPS, 0.1)
    next_t = 0.0
    next_card_t = 0.0
    last_level: str | None = None
    cards = 0
    index = -1
    t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index += 1
            t = index / fps
            if t + 1e-6 < next_t:
                continue
            next_t += step
            tracking.now = t
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            vision = predict_image(image, annotate=False, video=tracking)
            payload = assemble(vision, vibration, machine=machine, actuate=False)
            level = (payload.get("fusion") or {}).get("label") or ""
            if t >= next_card_t or level != last_level:
                detail = card_detail(payload, vision, source="upload")
                detail["declared"] = dict(recording.declared or {})
                ok_jpeg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                store_card(recording, cards, t, detail, encoded.tobytes() if ok_jpeg else b"")
                cards += 1
                next_card_t = t + CARD_STEP_S
                last_level = level
    finally:
        cap.release()
    if cards == 0:
        raise ValueError("The video has no readable frames")
    recording.duration_s = t
    recording.fps = fps
    recording.problem_count = recording.minutes.filter(level__in=PROBLEM_LEVELS).count()
    recording.render_status = "ready"
    recording.save(update_fields=["duration_s", "fps", "problem_count", "render_status"])


def enqueue(pk: int) -> None:
    """Queue analysis on Celery, or a local thread when the broker is unavailable."""
    from monitoring.tasks import analyze_upload_recording

    pk = int(pk)
    try:
        analyze_upload_recording.delay(pk)
        logger.info("Queued upload analysis %s on Celery", pk)
        return
    except Exception as exc:  # noqa: BLE001 — broker down, eager misconfig, etc.
        logger.warning("Celery unavailable for upload %s (%s); using a local thread", pk, exc)

    def _run() -> None:
        from monitoring.models import Recording

        close_old_connections()
        try:
            analyze_recording(pk)
        except Exception:
            logger.exception("Video analysis %s failed", pk)
            Recording.objects.filter(pk=pk).update(render_status="failed")
        finally:
            close_old_connections()

    threading.Thread(target=_run, name=f"sams-video-{pk}", daemon=True).start()


def resume_pending() -> None:
    """Queue uploads a restart left unanalysed."""
    from monitoring.models import Recording

    close_old_connections()
    try:
        for pk in Recording.objects.filter(source="upload", render_status="pending").values_list("pk", flat=True):
            enqueue(pk)
    finally:
        close_old_connections()
