"""Save an analyzed workplace video and a log card at each sampled second."""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

from django.core.files.base import ContentFile

from monitoring.models import Recording, RecordingMinute
from monitoring.services.vision_inference import predict_image

logger = logging.getLogger("monitoring.recordings")

MAX_LOGS = 48
LOG_STEP_S = 2.0
PPE_KEYS = ("hard_hat", "vest", "gloves", "boots")
_DISTANCE_LABEL = {"tof": "sensor", "vision": "body"}


def log_line(detail: dict | None, level: str = "") -> str:
    """One honest sentence. Distance appears only for a sensor or full-body reading."""
    detail = detail or {}
    reason = str(detail.get("reason") or "").strip()
    if reason:
        return reason
    if not detail.get("worker_detected"):
        return "No person"
    parts: list[str] = []
    summary = detail.get("summary")
    if summary and summary not in {"—", "Uncertain", level, "No person"}:
        parts.append(str(summary))
    count = detail.get("person_count")
    try:
        count_n = int(count) if count is not None else 0
    except (TypeError, ValueError):
        count_n = 0
    if count_n > 0:
        parts.append("1 person" if count_n == 1 else f"{count_n} people")
    source = detail.get("distance_source")
    distance = detail.get("distance_m")
    if source in _DISTANCE_LABEL and distance is not None:
        try:
            parts.append(f"{float(distance):.1f} m {_DISTANCE_LABEL[source]}")
        except (TypeError, ValueError):
            pass
    return " · ".join(parts) if parts else (level or "Person")


def log_line_from_prediction(pred: dict | None, level: str = "") -> str:
    pred = pred or {}
    ppe = pred.get("ppe") if isinstance(pred.get("ppe"), dict) else {}
    return log_line(
        {
            "worker_detected": pred.get("worker_detected"),
            "summary": pred.get("ppe_summary") or ppe.get("summary") or pred.get("label"),
            "person_count": pred.get("person_count"),
            "distance_m": pred.get("worker_distance_est"),
            "distance_source": pred.get("distance_source"),
        },
        level,
    )


def _store_logs(recording: Recording, samples: list[tuple[float, object]]) -> int:
    """Write one card per sample, at that frame's own timestamp."""
    from monitoring.services.ppe_inference import reset_live_votes
    from monitoring.services.vision_inference import reset_tracks

    reset_tracks()
    reset_live_votes()
    judged = []
    for offset_s, frame in samples:
        row = _judge_frame(frame)
        row["offset_s"] = float(offset_s)
        judged.append(row)
    for mark in list(recording.minutes.all()):
        if mark.snapshot:
            mark.snapshot.delete(save=False)
        mark.delete()
    problems = 0
    for index, row in enumerate(judged):
        if row["level"] in {"CAUTION", "DANGER"}:
            problems += 1
        mark = RecordingMinute(
            recording=recording,
            minute=index,
            offset_s=row["offset_s"],
            level=row["level"],
            summary=row["summary"],
            detail=row["detail"],
        )
        if row["jpeg"]:
            mark.snapshot.save(f"m{index:03d}.jpg", ContentFile(row["jpeg"]), save=False)
        mark.save()
    recording.problem_count = problems
    recording.save(update_fields=["problem_count"])
    return len(judged)


def active_log_index(offsets: list[float], time_s: float) -> int | None:
    """The last saved sample at or just before this time in the clip."""
    chosen = None
    for index, offset in enumerate(offsets):
        if float(offset) <= float(time_s) + 0.05:
            chosen = index
    return chosen


def save_analyzed_video(video_bytes: bytes, filename: str) -> Recording:
    """Store the clean clip and queue the log plus the framed copy."""
    title = Path(filename or "clip").name
    recording = Recording.objects.create(title=title, render_status="pending")
    try:
        recording.video.save(_safe_name(title), ContentFile(video_bytes), save=True)
    except Exception:
        recording.delete()
        raise
    from monitoring.services.video_jobs import enqueue_recording

    enqueue_recording(recording.pk)
    return recording


def relog_recording(recording: Recording) -> int:
    """Rebuild an existing video's cards at the seconds they were sampled."""
    if not recording.video:
        return 0
    duration_s, fps, samples = _sample_logs(Path(recording.video.path).read_bytes())
    if duration_s:
        recording.duration_s = duration_s
    if fps:
        recording.fps = fps
    recording.save(update_fields=["duration_s", "fps"])
    return _store_logs(recording, samples)


def _needs_judge(recording: Recording) -> bool:
    minutes = list(recording.minutes.all())
    if not minutes:
        return True
    return not any(isinstance((mark.detail or {}).get("boxes"), list) for mark in minutes)


def finish_recording(pk: int) -> None:
    """Write the log cards and the framed video for one clean clip."""
    recording = Recording.objects.get(pk=pk)
    if not recording.video:
        recording.render_status = "failed"
        recording.save(update_fields=["render_status"])
        return
    try:
        if _needs_judge(recording):
            relog_recording(recording)
            recording.refresh_from_db()
        _save_framed(recording)
        recording.render_status = "ready"
        recording.save(update_fields=["render_status"])
    except Exception:
        logger.exception("Could not frame recording %s", pk)
        Recording.objects.filter(pk=pk).update(render_status="failed")


def _save_framed(recording: Recording) -> None:
    stem = Path(recording.video.name).stem or "clip"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        dest = Path(tmp.name)
    try:
        write_framed_video(recording, dest)
        if recording.framed:
            recording.framed.delete(save=False)
        with dest.open("rb") as handle:
            recording.framed.save(f"{stem}-framed.mp4", ContentFile(handle.read()), save=True)
    finally:
        dest.unlink(missing_ok=True)


def write_framed_video(recording: Recording, dest: Path) -> None:
    """Burn the saved boxes and the log line into a copy of the clean clip."""
    import cv2

    cap = cv2.VideoCapture(recording.video.path)
    if not cap.isOpened():
        raise ValueError("Could not open the clean video")
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise ValueError("Clean video has no frames")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or float(recording.fps or 0) or 25.0
    marks = list(recording.minutes.all())
    offsets = [float(mark.offset_s) for mark in marks]
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        frame = first
        index = 0
        while frame is not None:
            view = frame[:height, :width].copy()
            chosen = active_log_index(offsets, index / fps)
            if chosen is not None:
                _paint_log(view, marks[chosen], width, height)
            proc.stdin.write(view.tobytes())
            ok, frame = cap.read()
            if not ok:
                frame = None
            index += 1
        proc.stdin.close()
        proc.stdin = None
        error = proc.stderr.read().decode("utf-8", errors="replace")
        code = proc.wait(timeout=120)
        if code != 0:
            raise RuntimeError(error[-500:] or f"ffmpeg exited {code}")
    finally:
        cap.release()
        if proc.poll() is None:
            proc.kill()


def _paint_log(frame, mark: RecordingMinute, width: int, height: int) -> None:
    import cv2

    detail = mark.detail or {}
    source_w = float(detail.get("frame_w") or 0) or width
    source_h = float(detail.get("frame_h") or 0) or height
    sx = width / source_w
    sy = height / source_h
    for box in detail.get("boxes") or []:
        xyxy = box.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v) * (sx if i % 2 == 0 else sy)) for i, v in enumerate(xyxy[:4])]
        if box.get("kind") == "gear":
            color = (42, 50, 194) if box.get("missing") else (69, 138, 43)
            label = str(box.get("label") or "ppe")
        else:
            color = (0, 161, 230)
            label = f"person {box.get('track_id')}"
            if box.get("distance_m") is not None:
                label += f" {float(box['distance_m']):.1f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    line = f"{mark.level}  {mark.summary}".strip()
    if not line:
        return
    bar = max(28, height // 18)
    cv2.rectangle(frame, (0, height - bar), (width, height), (22, 22, 22), -1)
    cv2.putText(
        frame,
        line[:90],
        (8, height - max(8, bar // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _safe_name(name: str) -> str:
    cleaned = Path(name).name.replace("\x00", "")
    return cleaned or "clip.mp4"


def _log_times(duration_s: float) -> list[float]:
    """Sample across the clip. Each time keeps its real seconds."""
    if duration_s <= 0:
        return [0.0]
    step = LOG_STEP_S
    if int(duration_s // step) + 1 > MAX_LOGS:
        step = duration_s / MAX_LOGS
    times: list[float] = []
    cursor = 0.0
    while cursor < duration_s - 0.05 and len(times) < MAX_LOGS:
        times.append(round(cursor, 3))
        cursor += step
    return times or [0.0]


def _sample_logs(video_bytes: bytes) -> tuple[float, float, list[tuple[float, object]]]:
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Could not open video for the log")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total / fps) if total and fps else 0.0
        samples: list[tuple[float, object]] = []
        for offset_s in _log_times(duration_s):
            cap.set(cv2.CAP_PROP_POS_MSEC, offset_s * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                samples.append((offset_s, frame))
        cap.release()
    if not samples:
        raise ValueError("Video has no readable frames")
    return duration_s, fps, samples


def stored_boxes(pred: dict | None) -> list[dict]:
    """Person and gear corners from one judged frame, in that frame's pixels."""
    pred = pred or {}
    boxes: list[dict] = []
    for det in pred.get("detections") or []:
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            continue
        source = det.get("distance_source")
        distance = det.get("distance_m") if source in _DISTANCE_LABEL else None
        try:
            distance_m = None if distance is None else round(float(distance), 3)
        except (TypeError, ValueError):
            distance_m = None
        boxes.append(
            {
                "kind": "person",
                "xyxy": [round(float(v), 1) for v in xyxy[:4]],
                "track_id": det.get("track_id"),
                "is_primary": bool(det.get("is_primary")),
                "distance_m": distance_m,
            }
        )
    ppe = pred.get("ppe") if isinstance(pred.get("ppe"), dict) else {}
    for det in ppe.get("detections") or []:
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            continue
        boxes.append(
            {
                "kind": "gear",
                "xyxy": [round(float(v), 1) for v in xyxy[:4]],
                "track_id": det.get("track_id"),
                "label": str(det.get("label") or "ppe"),
                "missing": bool(det.get("missing")),
            }
        )
    return boxes


def _person_row(person: dict) -> dict:
    from monitoring.services.control_center import classify

    ppe = person.get("ppe") if isinstance(person.get("ppe"), dict) else {}
    items = []
    for key in PPE_KEYS:
        item = (ppe.get("items") or {}).get(key) or {}
        items.append(
            {
                "key": key,
                "label": item.get("label") or key.replace("_", " "),
                "worn": ppe.get(key),
                "status": item.get("status") or "—",
            }
        )
    source = person.get("distance_source")
    distance = person.get("distance_m") if source in _DISTANCE_LABEL else None
    situation = classify(
        {
            "worker_detected": True,
            "distance_source": source if distance is not None else None,
            "worker_distance_est": distance,
            "confidence": person.get("confidence"),
            "ppe": ppe,
        },
        None,
        machine_on=False,
    )
    return {
        "id": person.get("id"),
        "selected": bool(person.get("selected")),
        "distance_m": distance,
        "distance_source": source if distance is not None else None,
        "summary": person.get("summary"),
        "ppe": items,
        "level": situation["label"],
        "reason": situation["reason"],
    }


def _judge_frame(frame) -> dict:
    import cv2
    from PIL import Image

    from monitoring.services.control_center import classify

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    pred = predict_image(image, track=True)
    frame_w, frame_h = image.size
    people = [_person_row(person) for person in (pred.get("people") or [])]
    ranks = {"NORMAL": 0, "DEGRADED": 1, "CAUTION": 2, "DANGER": 3}
    if people:
        lead = max(people, key=lambda row: (ranks.get(row["level"], 0), row.get("selected")))
        level = lead["level"]
        reason = lead["reason"]
    else:
        situation = classify(pred, None, machine_on=False)
        level = situation["label"]
        reason = situation["reason"]
    ppe = pred.get("ppe") if isinstance(pred.get("ppe"), dict) else {}
    items = []
    for key in PPE_KEYS:
        item = (ppe.get("items") or {}).get(key) or {}
        worn = ppe.get(key)
        items.append(
            {
                "key": key,
                "label": item.get("label") or key.replace("_", " "),
                "worn": worn,
                "status": item.get("status") or "—",
            }
        )
    source = pred.get("distance_source")
    distance = pred.get("worker_distance_est") if source in _DISTANCE_LABEL else None
    if source not in _DISTANCE_LABEL:
        source = None
    detail = {
        "level": level,
        "reason": reason,
        "summary": ppe.get("summary") or pred.get("ppe_summary") or pred.get("label"),
        "vision": pred.get("label"),
        "confidence": pred.get("confidence"),
        "worker_detected": bool(pred.get("worker_detected")),
        "person_count": pred.get("person_count"),
        "distance_m": distance,
        "distance_source": source,
        "ppe": items,
        "people": people,
        "machine_on": False,
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "boxes": stored_boxes(pred),
    }
    summary = log_line(detail, level)
    jpeg = _jpeg_bytes(pred.get("annotated_thumbnail"))
    return {"level": level, "summary": str(summary)[:240], "detail": detail, "jpeg": jpeg}


def _jpeg_bytes(data_url: str | None) -> bytes:
    if not data_url or "," not in data_url:
        return b""
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except Exception:  # noqa: BLE001
        logger.warning("Could not decode a minute snapshot")
        return b""
