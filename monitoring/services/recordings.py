"""One-sentence log lines for live decisions and archived log cards."""

from __future__ import annotations

from pathlib import Path

_DISTANCE_LABEL = {"tof": "sensor", "vision": "body"}


def log_line(detail: dict | None, level: str = "") -> str:
    """One honest sentence. Distance appears only for a sensor or network reading."""
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


CARD_STEP_S = 2.0
PROBLEM_LEVELS = {"CAUTION", "DANGER"}


def ppe_items(ppe: dict | None) -> list[dict]:
    items = (ppe or {}).get("items") or {}
    return [
        {
            "key": key,
            "label": item.get("label") or key.replace("_", " "),
            "worn": item.get("worn"),
            "status": item.get("status") or "—",
        }
        for key, item in items.items()
    ]


def _xyxy(box) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    return [round(float(v), 1) for v in box[:4]]


def _keypoints(points) -> list[list[float]] | None:
    if not isinstance(points, list) or not points:
        return None
    return [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in points if len(p) >= 3]


def boxes_from_prediction(vision: dict | None) -> list[dict]:
    """Person, gear, fire/smoke and fall boxes of one frame, in that frame's pixels."""
    vision = vision or {}
    boxes: list[dict] = []
    for det in vision.get("detections") or []:
        xyxy = _xyxy(det.get("xyxy"))
        if xyxy is None:
            continue
        source = det.get("distance_source")
        boxes.append(
            {
                "kind": "person",
                "xyxy": xyxy,
                "track_id": det.get("track_id"),
                "is_primary": bool(det.get("is_primary")),
                "distance_m": det.get("distance_m") if source in _DISTANCE_LABEL else None,
                "fallen": bool(det.get("fallen")),
                "keypoints": _keypoints(det.get("keypoints")),
            }
        )
    ppe = vision.get("ppe") if isinstance(vision.get("ppe"), dict) else {}
    for det in ppe.get("detections") or []:
        xyxy = _xyxy(det.get("xyxy"))
        if xyxy is not None:
            boxes.append(
                {
                    "kind": "gear",
                    "xyxy": xyxy,
                    "track_id": det.get("track_id"),
                    "label": str(det.get("label") or "ppe"),
                    "missing": bool(det.get("missing")),
                }
            )
    hazards = vision.get("hazards") if isinstance(vision.get("hazards"), dict) else {}
    for det in hazards.get("detections") or []:
        xyxy = _xyxy(det.get("xyxy"))
        if xyxy is not None:
            boxes.append(
                {
                    "kind": "hazard",
                    "xyxy": xyxy,
                    "label": str(det.get("label") or "hazard"),
                    "confidence": round(float(det.get("confidence") or 0), 3),
                }
            )
    for det in vision.get("lone_falls") or []:
        xyxy = _xyxy(det.get("xyxy"))
        if xyxy is not None:
            boxes.append({"kind": "fall", "xyxy": xyxy, "fallen": True})
    return boxes


def card_detail(payload: dict, vision: dict | None, *, source: str) -> dict:
    """Everything one analysed frame produced, as an Archive card keeps it.

    ``features`` are the fusion inputs exactly as fusion saw them, so a kept
    card is a training row for notebook 03.
    """
    vision = vision or {}
    fusion = payload.get("fusion") or {}
    features = payload.get("features") or {}
    label = fusion.get("label") or ""
    reason = fusion.get("reason") or ""
    people = []
    for person in vision.get("people") or []:
        selected = bool(person.get("selected"))
        people.append(
            {
                "id": person.get("id"),
                "selected": selected,
                "distance_m": person.get("distance_m"),
                "distance_source": person.get("distance_source"),
                "approaching": person.get("approaching"),
                "fallen": bool(person.get("fallen")),
                "summary": person.get("summary"),
                "ppe": ppe_items(person.get("ppe")),
                "level": label if selected else "",
                "reason": reason if selected else "",
            }
        )
    vibration = {k: v for k, v in (payload.get("vibration") or {}).items() if k != "series"}
    return {
        "level": label,
        "reason": reason,
        "summary": vision.get("ppe_summary") or vision.get("label"),
        "vision": vision.get("label"),
        "confidence": vision.get("confidence"),
        "worker_detected": bool(vision.get("worker_detected")),
        "person_count": vision.get("person_count"),
        "distance_m": features.get("worker_distance"),
        "distance_source": vision.get("distance_source") if "worker_distance" in features else None,
        "ppe": ppe_items(vision.get("ppe")),
        "people": people,
        "frame_w": int(vision.get("frame_w") or 0),
        "frame_h": int(vision.get("frame_h") or 0),
        "boxes": boxes_from_prediction(vision),
        "fire_score": vision.get("fire_score"),
        "smoke_score": vision.get("smoke_score"),
        "fall_detected": vision.get("fall_detected"),
        "features": features,
        "fusion": {key: fusion.get(key) for key in ("label", "confidence", "probabilities", "model_id", "missing")},
        "vibration": vibration or None,
        "machine": payload.get("machine"),
        "source": source,
    }


def store_card(recording, index: int, offset_s: float, detail: dict, jpeg: bytes):
    """One RecordingMinute with the clean frame as its snapshot."""
    from django.core.files.base import ContentFile

    from monitoring.models import RecordingMinute

    mark = RecordingMinute(
        recording=recording,
        minute=index,
        offset_s=round(float(offset_s), 3),
        level=detail["level"],
        summary=log_line(detail, detail["level"])[:240],
        detail=detail,
    )
    if jpeg:
        mark.snapshot.save(f"r{recording.pk}_m{index:04d}.jpg", ContentFile(jpeg), save=False)
    mark.save()
    return mark


def encode_clip(frames: list[tuple[float, bytes | Path]], dest, *, fps: float = 5.0) -> float:
    """Write timed JPEG frames (bytes, or files on disk) as an H.264 mp4 on the real clock. Returns the duration in seconds.

    Each frame stays on screen until the next one's time, so a card's offset
    is the same second in the video.
    """
    import io

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    if not frames:
        raise ValueError("No frames to encode")
    frames = sorted(frames, key=lambda item: item[0])
    end_t = frames[-1][0] + 1.0 / fps
    size = None
    writer = imageio.get_writer(str(dest), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2)
    try:
        index = 0
        tick = 0
        shown, pixels = -1, None
        while tick / fps < end_t:
            t = tick / fps
            while index + 1 < len(frames) and frames[index + 1][0] <= t:
                index += 1
            if index != shown:
                data = frames[index][1]
                raw = data.read_bytes() if isinstance(data, Path) else data
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                if size is None:
                    size = (image.width - image.width % 2, image.height - image.height % 2)
                if image.size != size:
                    image = image.resize(size)
                shown, pixels = index, np.asarray(image)
            writer.append_data(pixels)
            tick += 1
    finally:
        writer.close()
    return end_t
