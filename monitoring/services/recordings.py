"""Analyze an uploaded workplace video with every vision model.

The clip is read once at ``ANALYSIS_FPS``. Each analyzed frame goes through
person tracking (the video's own PPE tracker), pose joints, gear on the
body, distance, fire/smoke, the fall detector checked against the pose,
and, once a second, the scene behaviour classifier. Every ``LOG_STEP_S``
seconds a log card keeps the worst frame of that window. The framed copy
draws the nearest analyzed frame on every frame of the clip.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from django.core.files.base import ContentFile

from monitoring.models import Recording, RecordingMinute
from monitoring.services.vision_inference import predict_image

logger = logging.getLogger("monitoring.recordings")

MAX_LOGS = 48
LOG_STEP_S = 2.0
ANALYSIS_FPS = float(os.environ.get("SAMS_VIDEO_ANALYSIS_FPS", "4"))
HAZARD_EVERY_S = float(os.environ.get("SAMS_VIDEO_HAZARD_EVERY_S", "0.5"))
FALL_EVERY_S = float(os.environ.get("SAMS_VIDEO_FALL_EVERY_S", "0.5"))
BEHAVIOUR_EVERY_S = 1.0
BEHAVIOUR_SHOW_PROB = 0.5
HAZARD_SHARE = 0.4
FALL_CHECKS = 3
FALL_MIN_HITS = 2
PPE_KEYS = ("hard_hat", "vest", "gloves", "boots")
_DISTANCE_LABEL = {"tof": "sensor", "vision": "body"}
_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)
_LEVEL_BGR = {
    "DANGER": (42, 50, 194),
    "CAUTION": (0, 161, 230),
    "DEGRADED": (120, 120, 120),
    "NORMAL": (69, 138, 43),
}


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
    """Analyze the clip again and rewrite its cards. Returns the card count."""
    if not recording.video:
        return 0
    analysis = analyze_clip(Path(recording.video.path), scope=f"rec{recording.pk}:")
    _store_cards(recording, analysis)
    return len(analysis.cards)


def finish_recording(pk: int) -> None:
    """Write the log cards and the framed video for one clean clip."""
    recording = Recording.objects.get(pk=pk)
    if not recording.video:
        recording.render_status = "failed"
        recording.save(update_fields=["render_status"])
        return
    try:
        analysis = analyze_clip(Path(recording.video.path), scope=f"rec{recording.pk}:")
        _store_cards(recording, analysis)
        _save_framed(recording, analysis)
        recording.render_status = "ready"
        recording.save(update_fields=["render_status"])
    except Exception:
        logger.exception("Could not frame recording %s", pk)
        Recording.objects.filter(pk=pk).update(render_status="failed")


@dataclass
class FrameResult:
    t: float
    level: str
    reason: str
    base_level: str
    base_reason: str
    boxes: list[dict]
    pred: dict
    fallen_ids: list[Any]
    behaviour: dict


@dataclass
class Card:
    offset_s: float
    level: str
    summary: str
    detail: dict
    jpeg: bytes


@dataclass
class ClipAnalysis:
    duration_s: float
    fps: float
    width: int
    height: int
    frames: list[FrameResult] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)


def analyze_clip(path: Path, *, scope: str = "clip:") -> ClipAnalysis:
    """Every vision model over the clip, with one tracker for the whole video."""
    import cv2
    from PIL import Image

    from monitoring.services import behaviour_inference as behaviour
    from monitoring.services import ppe_inference as ppe
    from monitoring.services.person_tracker import IdKeeper, MotionLog

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video for analysis")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = total / fps if total else 0.0
    card_times = _log_times(duration_s)
    analysis = ClipAnalysis(duration_s=duration_s, fps=fps, width=width, height=height)

    ppe_model = ppe.load_private_model() if ppe.weights_available() else None
    motion = MotionLog(history_s=1.5)
    keeper = IdKeeper()
    window = behaviour.BehaviourWindow(size=3)
    fall_checks: deque[set[Any]] = deque(maxlen=FALL_CHECKS)
    step_s = 1.0 / max(ANALYSIS_FPS, 0.1)
    next_t = 0.0
    next_behaviour_t = 0.0
    next_hazard_t = 0.0
    next_fall_t = 0.0
    current: list[tuple[FrameResult, Any]] = []
    last_hazard_boxes: list[dict] = []
    last_fall_boxes: list[dict] = []
    card_index = 0
    index = -1
    ppe.drop_votes(scope)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index += 1
            t = index / fps
            if t + 1e-6 < next_t:
                continue
            next_t += step_s
            while card_index + 1 < len(card_times) and t >= card_times[card_index + 1]:
                if current:
                    analysis.cards.append(_make_card(card_times[card_index], current))
                current = []
                card_index += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if t + 1e-6 >= next_behaviour_t:
                window.add(behaviour.classify(image))
                next_behaviour_t += BEHAVIOUR_EVERY_S
            run_hazards = t + 1e-6 >= next_hazard_t
            if run_hazards:
                next_hazard_t += HAZARD_EVERY_S
            run_falls = t + 1e-6 >= next_fall_t
            if run_falls:
                next_fall_t += FALL_EVERY_S
            pred = predict_image(
                image,
                id_keeper=keeper,
                ppe_model=ppe_model,
                motion=motion,
                now=t,
                vote_scope=scope,
                annotate=False,
                run_hazards=run_hazards,
                run_falls=run_falls,
            )
            result = _judge(t, pred, window.current(), fall_checks)
            if run_hazards:
                last_hazard_boxes = [box for box in result.boxes if box["kind"] == "hazard"]
            else:
                result.boxes.extend(last_hazard_boxes)
            if run_falls:
                last_fall_boxes = [box for box in result.boxes if box["kind"] == "fall" or box.get("fallen")]
            elif result.fallen_ids:
                present = {box.get("track_id") for box in result.boxes if box["kind"] == "person"}
                result.boxes.extend(
                    box for box in last_fall_boxes if box["kind"] == "fall" or box.get("track_id") not in present
                )
            analysis.frames.append(result)
            current.append((result, frame))
        if current:
            analysis.cards.append(_make_card(card_times[card_index], current))
    finally:
        cap.release()
        ppe.drop_votes(scope)
    if not analysis.frames:
        raise ValueError("Video has no readable frames")
    if not duration_s:
        analysis.duration_s = analysis.frames[-1].t
    return analysis


def _judge(t: float, pred: dict, behaviour: dict, fall_checks: deque) -> FrameResult:
    """One analyzed frame: per-person fusion level, raised by scene findings.

    A fall counts once ``FALL_MIN_HITS`` of the last ``FALL_CHECKS`` fall
    checks found somebody down, so one stray box never raises it and an id
    that changes mid-fall does not reset it.
    """
    from monitoring.services.control_center import LEVEL_RANK, apply_scene_events, scene_events

    if pred.get("falls_ran", True):
        seen_down: set[Any] = {person["id"] for person in pred.get("people") or [] if person.get("fallen")}
        if pred.get("lone_falls"):
            seen_down.add(None)
        fall_checks.append(seen_down)
    confirmed: list[Any] = []
    down_now: set[Any] = set()
    if sum(1 for check in fall_checks if check) >= FALL_MIN_HITS:
        confirmed = sorted(set().union(*fall_checks), key=lambda k: (k is None, k or 0))
        down_now = set(fall_checks[-1])

    people = [_person_row(person, list(down_now)) for person in pred.get("people") or []]
    if people:
        lead = max(people, key=lambda row: (LEVEL_RANK.get(row["level"], 0), row.get("selected")))
        level, reason = lead["level"], lead["reason"]
    else:
        from monitoring.services.control_center import classify

        situation = classify(pred, None, machine_on=False)
        level, reason = situation["label"], situation["reason"]
    if people and all(row["fallen"] for row in people):
        base_level, base_reason = "NORMAL", ""
    else:
        base_level, base_reason = level, reason
    level, reason = apply_scene_events(level, reason, scene_events(fallen_ids=confirmed, behaviour=behaviour))
    pred["people_rows"] = people
    boxes = stored_boxes(pred)
    for box in boxes:
        if box["kind"] == "person":
            box["fallen"] = box.get("track_id") in down_now
    boxes = [box for box in boxes if box["kind"] != "fall" or None in down_now]
    return FrameResult(
        t=t,
        level=level,
        reason=reason,
        base_level=base_level,
        base_reason=base_reason,
        boxes=boxes,
        pred=pred,
        fallen_ids=confirmed,
        behaviour=behaviour,
    )


def _make_card(offset_s: float, frames: list[tuple[FrameResult, Any]]) -> Card:
    """One log card for a window: fire and smoke must hold for part of it."""
    from monitoring.services.control_center import LEVEL_RANK, apply_scene_events, scene_events

    results = [result for result, _frame in frames]
    scanned = [r for r in results if r.pred.get("hazards_ran", True)]

    def share(key: str) -> float:
        if not scanned:
            return 0.0
        return sum(1 for r in scanned if float(r.pred.get(key) or 0) > 0) / len(scanned)

    fire = share("fire_score") >= HAZARD_SHARE
    smoke = share("smoke_score") >= HAZARD_SHARE
    fallen: list[Any] = []
    for result in results:
        for pid in result.fallen_ids:
            if pid not in fallen:
                fallen.append(pid)
    behaviour = results[-1].behaviour

    def weight(item: tuple[FrameResult, Any]) -> tuple:
        result = item[0]
        return (
            bool(result.fallen_ids),
            fire and float(result.pred.get("fire_score") or 0),
            smoke and float(result.pred.get("smoke_score") or 0),
            LEVEL_RANK.get(result.level, 0),
            int(result.pred.get("person_count") or 0),
        )

    rep, rep_frame = max(frames, key=weight)
    pred = rep.pred
    rows = [row for row in pred.get("people_rows") or [] if not row.get("fallen")]
    base_level = max((row["level"] for row in rows), key=lambda lv: LEVEL_RANK.get(lv, 0), default=None)
    lead_row = next((row for row in rows if row["level"] == base_level), None)
    base_reason = lead_row["reason"] if lead_row else ""
    if lead_row and base_level == "DEGRADED" and lead_row.get("distance_m") is None:
        base_reason = f"No distance for person {lead_row['id']}"
    if base_level is None:
        base_level = rep.base_level
        base_reason = rep.base_reason
    events = scene_events(fallen_ids=fallen, fire=fire, smoke=smoke, behaviour=behaviour)
    level, _ = apply_scene_events(base_level, base_reason, events)

    ppe = pred.get("ppe") if isinstance(pred.get("ppe"), dict) else {}
    source = pred.get("distance_source")
    distance = pred.get("worker_distance_est") if source in _DISTANCE_LABEL else None
    detail = {
        "level": level,
        "reason": base_reason,
        "summary": ppe.get("summary") or pred.get("ppe_summary") or pred.get("label"),
        "vision": pred.get("label"),
        "confidence": pred.get("confidence"),
        "worker_detected": bool(pred.get("worker_detected")),
        "person_count": pred.get("person_count"),
        "distance_m": distance,
        "distance_source": source if source in _DISTANCE_LABEL else None,
        "ppe": _ppe_items(ppe),
        "people": pred.get("people_rows") or [],
        "machine_on": False,
        "frame_w": int(pred.get("frame_w") or 0),
        "frame_h": int(pred.get("frame_h") or 0),
        "boxes": rep.boxes,
        "frame_t": round(rep.t, 3),
        "fire_score": round(max(float(r.pred.get("fire_score") or 0) for r in results), 3),
        "smoke_score": round(max(float(r.pred.get("smoke_score") or 0) for r in results), 3),
        "fallen_ids": [pid for pid in fallen if pid is not None],
        "behaviour": behaviour,
        "events": [{"level": lv, "text": text} for lv, text in events],
    }
    seen = log_line({**detail, "reason": ""}, base_level)
    parts = [part for part in (seen, base_reason) if part and part != base_level]
    base_line = " · ".join(parts) or log_line(detail, base_level)
    summary = " · ".join([text for _lv, text in events] + [base_line]) if events else base_line
    if events:
        detail["reason"] = summary
    jpeg = _snapshot(rep_frame, rep.boxes, level, summary, behaviour)
    return Card(offset_s=float(offset_s), level=level, summary=summary[:240], detail=detail, jpeg=jpeg)


def _store_cards(recording: Recording, analysis: ClipAnalysis) -> None:
    for mark in list(recording.minutes.all()):
        if mark.snapshot:
            mark.snapshot.delete(save=False)
        mark.delete()
    problems = 0
    for index, card in enumerate(analysis.cards):
        if card.level in {"CAUTION", "DANGER"}:
            problems += 1
        mark = RecordingMinute(
            recording=recording,
            minute=index,
            offset_s=card.offset_s,
            level=card.level,
            summary=card.summary,
            detail=card.detail,
        )
        if card.jpeg:
            mark.snapshot.save(f"m{index:03d}.jpg", ContentFile(card.jpeg), save=False)
        mark.save()
    recording.duration_s = analysis.duration_s
    recording.fps = analysis.fps
    recording.problem_count = problems
    recording.save(update_fields=["duration_s", "fps", "problem_count"])


def _save_framed(recording: Recording, analysis: ClipAnalysis) -> None:
    stem = Path(recording.video.name).stem or "clip"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        dest = Path(tmp.name)
    try:
        write_framed_video(Path(recording.video.path), dest, analysis)
        if recording.framed:
            recording.framed.delete(save=False)
        with dest.open("rb") as handle:
            recording.framed.save(f"{stem}-framed.mp4", ContentFile(handle.read()), save=True)
    finally:
        dest.unlink(missing_ok=True)


def write_framed_video(source: Path, dest: Path, analysis: ClipAnalysis) -> None:
    """Draw the nearest analyzed frame and the active card on every frame."""
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Could not open the clean video")
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise ValueError("Clean video has no frames")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or analysis.fps or 25.0
    frame_times = [result.t for result in analysis.frames]
    card_offsets = [card.offset_s for card in analysis.cards]
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dest),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        frame = first
        index = 0
        while frame is not None:
            view = frame[:height, :width].copy()
            t = index / fps
            shown = active_log_index(frame_times, t)
            if shown is not None:
                _paint_boxes(view, analysis.frames[shown].boxes)
            card = active_log_index(card_offsets, t)
            if card is not None:
                chosen = analysis.cards[card]
                _paint_banner(view, chosen.level, chosen.summary, chosen.detail.get("behaviour"))
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


def _snapshot(frame, boxes: list[dict], level: str, summary: str, behaviour: dict | None) -> bytes:
    import cv2

    view = frame.copy()
    _paint_boxes(view, boxes)
    _paint_banner(view, level, summary, behaviour)
    ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes() if ok else b""


def _label(frame, text: str, x: int, y: int, color) -> None:
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(0, y - th - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _paint_boxes(frame, boxes: list[dict]) -> None:
    import cv2

    for box in boxes or []:
        xyxy = box.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in xyxy[:4]]
        kind = box.get("kind")
        if kind == "gear":
            color = (42, 50, 194) if box.get("missing") else (69, 138, 43)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _label(frame, str(box.get("label") or "ppe"), x1, y1, color)
            continue
        if kind == "hazard":
            color = (0, 69, 255) if box.get("label") == "fire" else (140, 140, 140)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _label(frame, f"{box.get('label')} {float(box.get('confidence') or 0):.0%}", x1, y1, color)
            continue
        fallen = bool(box.get("fallen"))
        color = (42, 50, 194) if fallen else (0, 161, 230)
        _paint_skeleton(frame, box.get("keypoints"), color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if fallen else 2)
        if kind == "fall":
            label = "FALLEN"
        else:
            label = f"person {box.get('track_id')}"
            if box.get("distance_m") is not None:
                label += f" {float(box['distance_m']):.1f}m"
            if fallen:
                label += " FALLEN"
        _label(frame, label, x1, y1, color)


def _paint_skeleton(frame, keypoints, color) -> None:
    import cv2

    if not isinstance(keypoints, list) or len(keypoints) < 17:
        return
    points = [
        (int(p[0]), int(p[1])) if isinstance(p, list) and len(p) >= 3 and float(p[2]) >= 0.3 else None
        for p in keypoints
    ]
    for a, b in _SKELETON:
        if points[a] and points[b]:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
    for point in points:
        if point:
            cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)


def _paint_banner(frame, level: str, summary: str, behaviour: dict | None) -> None:
    import cv2

    height, width = frame.shape[:2]
    lines = [f"{level}  {summary.replace(' · ', ' | ')}".strip()[:110]]
    if behaviour and behaviour.get("label") and (behaviour.get("unsafe") or float(behaviour.get("prob") or 0) >= BEHAVIOUR_SHOW_PROB):
        tag = "UNSAFE" if behaviour.get("unsafe") else "Behaviour"
        lines.append(f"{tag}: {behaviour['label']} {float(behaviour.get('prob') or 0):.0%}")
    bar = max(28, height // 18)
    total = bar * len(lines)
    cv2.rectangle(frame, (0, height - total), (width, height), (22, 22, 22), -1)
    cv2.rectangle(frame, (0, height - total), (10, height), _LEVEL_BGR.get(level, (200, 200, 200)), -1)
    for row, text in enumerate(lines):
        y = height - total + bar * row + bar - max(8, bar // 3)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


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


def _round_keypoints(keypoints) -> list[list[float]] | None:
    if not isinstance(keypoints, list) or not keypoints:
        return None
    return [[round(float(v), 1) if i < 2 else round(float(v), 2) for i, v in enumerate(point[:3])] for point in keypoints]


def stored_boxes(pred: dict | None) -> list[dict]:
    """Person, gear, fire/smoke, and fall corners from one judged frame, in that frame's pixels."""
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
                "fallen": bool(det.get("fallen")),
                "keypoints": _round_keypoints(det.get("keypoints")),
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
    hazards = pred.get("hazards") if isinstance(pred.get("hazards"), dict) else {}
    for det in hazards.get("detections") or []:
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            continue
        boxes.append(
            {
                "kind": "hazard",
                "xyxy": [round(float(v), 1) for v in xyxy[:4]],
                "label": str(det.get("label") or "hazard"),
                "confidence": round(float(det.get("confidence") or 0), 3),
            }
        )
    for det in pred.get("lone_falls") or []:
        boxes.append(
            {
                "kind": "fall",
                "xyxy": [round(float(v), 1) for v in det["xyxy"][:4]],
                "fallen": True,
                "keypoints": _round_keypoints(det.get("keypoints")),
                "confidence": round(float(det.get("confidence") or 0), 3),
            }
        )
    return boxes


def _ppe_items(ppe: dict) -> list[dict]:
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
    return items


def _person_row(person: dict, fallen_ids: list[Any] | None = None) -> dict:
    from monitoring.services.control_center import classify

    ppe = person.get("ppe") if isinstance(person.get("ppe"), dict) else {}
    source = person.get("distance_source")
    distance = person.get("distance_m") if source in _DISTANCE_LABEL else None
    approach = float(person.get("approach_mps") or 0.0)
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
        speed=abs(approach),
        approaching=bool(person.get("approaching")),
    )
    level, reason = situation["label"], situation["reason"]
    fallen = person.get("id") in (fallen_ids or [])
    if fallen:
        level, reason = "DANGER", "Fallen"
    return {
        "id": person.get("id"),
        "selected": bool(person.get("selected")),
        "distance_m": distance,
        "distance_source": source if distance is not None else None,
        "approaching": bool(person.get("approaching")),
        "fallen": fallen,
        "summary": person.get("summary"),
        "ppe": _ppe_items(ppe),
        "level": level,
        "reason": reason,
    }


def _jpeg_bytes(data_url: str | None) -> bytes:
    if not data_url or "," not in data_url:
        return b""
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except Exception:  # noqa: BLE001
        logger.warning("Could not decode a minute snapshot")
        return b""
