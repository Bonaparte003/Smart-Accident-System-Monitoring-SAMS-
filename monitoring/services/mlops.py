"""Model inventory and the training export built from archived videos."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from django.conf import settings

from monitoring.models import RecordingMinute
from monitoring.services.ppe_inference import ppe_items

SERVING = (
    ("Person", "The PPE checkpoint's person class", "models/vision/ppe/best.pt"),
    ("PPE", "Hard hat, vest, gloves, and safety shoes on each person", "models/vision/ppe/best.pt"),
    ("Pose", "Body joints inside each worker crop", "models/vision/pose/best.pt"),
    ("Distance", "Residual network on the pinhole prior, from each person's box and pose", "models/vision/distance/distance_nn.pt"),
    ("Hazard", "Fire and smoke boxes", "models/vision/hazard/best.pt"),
    ("Accident", "A fallen worker", "models/vision/accident/best.pt"),
    ("Vibration", "Labels each live accelerometer window NORMAL or ABNORMAL", "models/vibration/best/model.joblib"),
    ("Machine state", "Labels each live accelerometer window STOPPED or RUNNING", "models/machine_state/model.joblib"),
    ("Fusion", "Turns vision, distance, and vibration into an alert", "models/fusion/best_fusion_model/model.joblib"),
)



def model_cards() -> list[dict]:
    """Filesystem status of the weights the live system uses. Does not load them."""
    root = Path(settings.BASE_DIR)
    cards = []
    for name, role, relative in SERVING:
        path = root / relative
        size = path.stat().st_size if path.is_file() else 0
        if size >= 100 * 1024:
            size_label = f"{size / (1024 * 1024):.1f} MB"
        elif size:
            size_label = f"{max(1, round(size / 1024))} KB"
        else:
            size_label = ""
        cards.append(
            {
                "name": name,
                "role": role,
                "relative": relative,
                "ready": path.is_file(),
                "size_label": size_label,
            }
        )
    return cards


def export_zip(minutes) -> bytes:
    """Zip the clean source videos, clean frames, and labels.csv.

    The framed copies stay in the archive. Training uses the original pixels.
    """
    buffer = io.BytesIO()
    minute_list = list(minutes)
    ppe_keys = list(ppe_items())
    clean = _clean_frames(minute_list)
    written: set[int] = set()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        rows = io.StringIO()
        writer = csv.writer(rows)
        writer.writerow(
            [
                "recording",
                "minute",
                "offset_s",
                "video",
                "image",
                "level",
                "true_label",
                "train_status",
                "source",
                "summary",
                "worker_detected",
                "distance_m",
                *ppe_keys,
            ]
        )
        for mark in minute_list:
            recording = mark.recording
            video_name = ""
            if recording.pk not in written and recording.video:
                video_name = f"videos/r{recording.pk}_{Path(recording.video.name).name}"
                try:
                    archive.write(recording.video.path, video_name)
                    written.add(recording.pk)
                except Exception:
                    video_name = ""
            elif recording.pk in written and recording.video:
                video_name = f"videos/r{recording.pk}_{Path(recording.video.name).name}"
            image_name = ""
            jpeg = clean.get(mark.pk) or b""
            if not jpeg and mark.snapshot:
                try:
                    jpeg = mark.snapshot.read()
                except Exception:
                    jpeg = b""
                finally:
                    try:
                        mark.snapshot.close()
                    except Exception:
                        pass
            if jpeg:
                image_name = f"frames/r{mark.recording_id}_m{mark.minute:03d}.jpg"
                archive.writestr(image_name, jpeg)
            detail = mark.detail or {}
            worn = {item.get("key"): item.get("worn") for item in detail.get("ppe") or []}
            writer.writerow(
                [
                    recording.title,
                    mark.minute,
                    mark.offset_s,
                    video_name,
                    image_name,
                    mark.level,
                    mark.true_label,
                    mark.train_status,
                    recording.source,
                    mark.summary,
                    detail.get("worker_detected"),
                    detail.get("distance_m"),
                    *[worn.get(key) for key in ppe_keys],
                ]
            )
        archive.writestr("labels.csv", rows.getvalue())
        from monitoring.services.trials import write_datasets

        write_datasets(archive)
    return buffer.getvalue()


def _clean_frames(minutes) -> dict[int, bytes]:
    """JPEG of the clean clip at each sample. These are the pixels for a retrain."""
    import cv2

    by_recording: dict[int, list] = {}
    for mark in minutes:
        by_recording.setdefault(mark.recording_id, []).append(mark)
    frames: dict[int, bytes] = {}
    for group in by_recording.values():
        recording = group[0].recording
        if not recording.video:
            continue
        cap = cv2.VideoCapture(recording.video.path)
        if not cap.isOpened():
            continue
        try:
            for mark in group:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(mark.offset_s) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    frames[mark.pk] = buf.tobytes()
        finally:
            cap.release()
    return frames


def kept_minutes():
    return (
        RecordingMinute.objects.filter(train_status="kept")
        .select_related("recording")
        .order_by("recording_id", "minute")
    )
