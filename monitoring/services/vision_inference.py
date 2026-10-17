"""One frame through every vision model, joined per person.

Who is in the frame is the Construction-PPE person class. Pose runs inside
each person box and places joints for the distance network. A gear box or a
fall box belongs to the person box that holds its centre. Distance is the
time-of-flight reading for the person the beam points at, otherwise the
trained distance network. The fire/smoke detector runs on the whole frame.

Live frames keep the track id the PPE checkpoint's tracker assigns. Each
track's distance history gives its speed and approach velocity.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from PIL import Image, ImageDraw, ImageFont

from . import distance_estimation as distance
from . import fusion_inference as fusion
from . import hazard_inference as hazard
from . import ppe_inference as ppe
from .person_tracker import MotionLog

_lock = threading.Lock()
_pose_model = None
_pose_spec: dict[str, Any] = {}
_yolo_error: str | None = None


def _pose_weights_path() -> Path:
    custom = os.environ.get("SAMS_POSE_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    trained = Path(settings.BASE_DIR) / "models" / "vision" / "pose" / "best.pt"
    if trained.is_file():
        return trained
    return Path(settings.BASE_DIR) / "models" / "vision" / "yolov8n-pose.pt"


def _load_pose_spec(weights: Path) -> dict[str, Any]:
    meta_path = weights.parent / "metadata.json"
    if weights.parent.name != "pose" or not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return {}


def _pose_enabled() -> bool:
    return os.environ.get("SAMS_VISION_POSE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def yolo_available() -> bool:
    try:
        import ultralytics

        return True
    except Exception as exc:
        global _yolo_error
        _yolo_error = str(exc)
        return False


def get_torch_status() -> dict[str, Any]:
    """Vision model status for the health check."""
    ok = yolo_available()
    try:
        calib = distance.load_calib()
    except (OSError, ValueError) as exc:
        calib = {"error": str(exc)}
    return {
        "available": ok,
        "model_id": _model_id(),
        "detail": None if ok else (_yolo_error or "ultralytics is not installed."),
        "task": "person_detection+distance+ppe",
        "class_names": ["person"],
        "distance": {
            "method": "ppe_person_then_pose",
            "weights": distance.weights_path().is_file(),
            "pose_enabled": _pose_enabled(),
            "tof_near_m": calib.get("tof_near_m"),
            "fy_px": calib.get("fy_px"),
        },
        "ppe": ppe.get_status(),
        "hazard": hazard.get_status(),
    }


def _model_id() -> str:
    parts = [ppe.model_id()]
    pose_weights = _pose_weights_path()
    if pose_weights.is_file():
        parts.append(str(_load_pose_spec(pose_weights).get("model_id") or pose_weights.stem))
    served = distance.served_model().get("experiment_id")
    if served:
        parts.append(str(served))
    status = hazard.get_status()
    if status.get("available"):
        parts.append(str(status.get("model_id") or "hazard"))
    accident_status = status.get("accident") or {}
    if accident_status.get("available"):
        parts.append(str(accident_status.get("model_id") or "accident"))
    return "+".join(part for part in parts if part)


def get_pose_model():
    """Pose checkpoint that places COCO-17 joints inside each worker crop."""
    global _pose_model, _pose_spec
    if not _pose_enabled() or not yolo_available():
        return None
    if _pose_model is None:
        with _lock:
            if _pose_model is None:
                try:
                    from ultralytics import YOLO

                    weights = _pose_weights_path()
                    if not weights.is_file():
                        import logging

                        logging.getLogger("monitoring.vision").warning("pose weights missing at %s", weights)
                        return None
                    _pose_spec = _load_pose_spec(weights)
                    _pose_model = YOLO(str(weights))
                except Exception as exc:  # noqa: BLE001
                    import logging

                    logging.getLogger("monitoring.vision").warning("pose model unavailable: %s", exc)
                    return None
    return _pose_model


def _read_tof_m() -> float | None:
    from hardware.tof import get_tof

    return get_tof().read_metres()


def _pose_operating_point() -> dict[str, Any]:
    """Person cutoff and crop padding the pose notebook chose, when its metadata exists."""
    return dict(_pose_spec.get("operating_point") or {})


def _run_pose_frame(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    pose = get_pose_model()
    if pose is None:
        return None
    kwargs: dict[str, Any] = {"verbose": False}
    if _pose_spec.get("imgsz"):
        kwargs["imgsz"] = int(_pose_spec["imgsz"])
    if _pose_operating_point().get("person_conf") is not None:
        kwargs["conf"] = float(_pose_operating_point()["person_conf"])
    try:
        results = pose.predict(image, **kwargs)
    except Exception:
        return None
    if not results:
        return None
    r0 = results[0]
    kps_obj = getattr(r0, "keypoints", None)
    boxes = getattr(r0, "boxes", None)
    if kps_obj is None or boxes is None or len(kps_obj) == 0:
        return None
    xy = kps_obj.xy.cpu().numpy()
    conf = kps_obj.conf.cpu().numpy() if kps_obj.conf is not None else np.ones(xy.shape[:2])
    bboxes = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(bboxes))
    return xy, conf, bboxes, scores


def _draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    ppe_detections: list[dict[str, Any]] | None = None,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        color = "#e03121" if det.get("is_primary") else "#fab005"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        dist_m = det.get("distance_m")
        src = det.get("distance_source") or ""
        if dist_m is not None:
            label = f"{det['label']} {det['confidence']:.0%} · {dist_m:.1f}m"
            if src:
                label += f" [{src}]"
        else:
            label = f"{det['label']} {det['confidence']:.0%}"
        text_y = max(0, y1 - 14)
        if font is not None:
            bbox = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, text_y), label, fill="white", font=font)
        else:
            draw.text((x1, text_y), label, fill=color)

    # PPE boxes (worn = green, missing = red)
    for det in ppe_detections or []:
        x1, y1, x2, y2 = det["xyxy"]
        name = str(det.get("label") or "")
        missing = bool(det.get("missing")) or name.lower().startswith("no_") or name.lower().startswith("no-") or name.lower().startswith("no ")
        color = "#e03121" if missing else "#2b8a3e"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{name} {float(det['confidence']):.0%}"
        text_y = max(0, int(y1) - 12)
        if font is not None:
            bbox = draw.textbbox((x1, text_y), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, text_y), label, fill="white", font=font)
        else:
            draw.text((x1, text_y), label, fill=color)
    return annotated


_motion: MotionLog | None = None


def _motion_log() -> MotionLog:
    """The live motion log, over the window the fusion model's motion inputs were measured on."""
    global _motion
    if _motion is None:
        _motion = MotionLog(history_s=fusion.motion_window_s())
    return _motion


class VideoTracking:
    """Tracker state for one uploaded clip: its own PPE model instance and motion log on the clip's clock."""

    def __init__(self) -> None:
        self.ppe_model = ppe.load_private_model() if ppe.weights_available() and ppe.ppe_enabled() else None
        self.motion = MotionLog(history_s=fusion.motion_window_s())
        self.now = 0.0


def reset_tracks() -> None:
    """Drop live distance history when a camera session starts or stops."""
    _motion_log().reset()
    ppe.reset_tracker()
    hazard.reset()


def _crop_worker(image: Image.Image, box: list[float]):
    """Crop one worker box, padded as the pose notebook's operating point says."""
    width, height = image.size
    pad_frac = float(_pose_operating_point().get("crop_pad", 0.0))
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x = (x2 - x1) * pad_frac
    pad_y = (y2 - y1) * pad_frac
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)), (left, top)


def _pose_inside(image: Image.Image, box: list[float]) -> list[list[float]] | None:
    """Joints of the highest-scoring pose inside one worker box, in frame pixels."""
    cropped = _crop_worker(image, box)
    if cropped is None:
        return None
    crop, (ox, oy) = cropped
    pack = _run_pose_frame(crop)
    if pack is None:
        return None
    xy, kp_conf, _boxes, scores = pack
    best = int(np.argmax(scores))
    return [
        [float(xy[best, joint, 0]) + ox, float(xy[best, joint, 1]) + oy, float(kp_conf[best, joint])]
        for joint in range(xy.shape[1])
    ]


def _centre_in(box: list[float], person: list[float]) -> bool:
    cx = (float(box[0]) + float(box[2])) * 0.5
    cy = (float(box[1]) + float(box[3])) * 0.5
    return float(person[0]) <= cx <= float(person[2]) and float(person[1]) <= cy <= float(person[3])


def _area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _assign(person_boxes: list[list[float]], boxes: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Give each box to the person box holding its centre (the smallest such box when several do)."""
    owned: list[list[dict[str, Any]]] = [[] for _ in person_boxes]
    unowned: list[dict[str, Any]] = []
    for det in boxes:
        holders = [i for i, person in enumerate(person_boxes) if _centre_in(det["xyxy"], person)]
        if not holders:
            unowned.append(det)
            continue
        owned[min(holders, key=lambda i: _area(person_boxes[i]))].append(det)
    return owned, unowned


def _select_person(people: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The alert follows the person closest to the machine."""
    if not people:
        return None

    def rank(person: dict[str, Any]) -> tuple:
        distance_m = person.get("distance_m")
        return (distance_m if distance_m is not None else float("inf"), -float(person["confidence"]))

    return min(people, key=rank)


def _image_to_data_url(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def predict_image(
    image: Image.Image,
    *,
    live: bool = False,
    annotate: bool = True,
    video: VideoTracking | None = None,
) -> dict[str, Any]:
    """One frame through every vision model, joined per person.

    ``live`` tracks people with the shared tracker and reads the ToF sensor.
    ``video`` tracks them through an uploaded clip with its own tracker and
    the clip's clock, so it never disturbs a live session.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    tof_m = _read_tof_m() if live else None
    tracking = live or video is not None
    motion_log = video.motion if video is not None else _motion_log()
    now = video.now if video is not None else None
    workers, gear, blocked = ppe.detect_scene(
        rgb, track=tracking, model=video.ppe_model if video is not None else None
    )
    if blocked is not None:
        workers, gear = [], []
    person_boxes = [worker["box"] for worker in workers]
    gear_for, _loose_gear = _assign(person_boxes, gear)

    hazards = hazard.detect_hazards(rgb)
    if live:
        hazards = hazard.smooth_scores(hazards)
    falls = hazard.detect_falls(rgb)
    falls_for, lone_falls = _assign(person_boxes, falls.get("detections") or [])

    aimed_index = None
    for index, worker in enumerate(workers):
        if distance.aim_hits(worker["box"], (width, height)):
            if aimed_index is None or worker["score"] > workers[aimed_index]["score"]:
                aimed_index = index

    people: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    drawn_gear: list[dict[str, Any]] = []
    distance_pairs: list[dict[str, Any]] = []
    for index, worker in enumerate(workers):
        keypoints = _pose_inside(rgb, worker["box"])
        tracked = tracking and worker.get("track_id") is not None
        track_id = int(worker["track_id"]) if tracked else index + 1
        wearing = blocked if blocked is not None else ppe.assess_person(gear_for[index])
        estimate = distance.measure_person(
            keypoints,
            (width, height),
            box=worker["box"],
            tof_m=tof_m,
            sensor_aimed=index == aimed_index,
            det_conf=float(worker["score"]),
            kpt_conf=_pose_operating_point().get("kpt_conf"),
        )
        distance_m = estimate.distance_m if estimate else None
        source = estimate.source if estimate else None
        if estimate is not None and source == "tof" and estimate.features:
            # The sensor's metres label this person's body features: a site row for notebook 04.
            distance_pairs.append(
                {"features": estimate.features, "z0_m": estimate.z0_m, "distance_m": estimate.distance_m}
            )
        if tracked:
            motion_log.record_distance(track_id, distance_m, now=now)
            motion = motion_log.motion(track_id)
        else:
            motion = {"speed_mps": None, "approach_mps": None, "approaching": None, "samples": 0}
        fall_boxes = falls_for[index]
        fallen = bool(fall_boxes)
        people.append(
            {
                "id": track_id,
                "confidence": float(worker["score"]),
                "xyxy": worker["box"],
                "distance_m": distance_m,
                "distance_source": source,
                "summary": wearing.get("summary"),
                "ppe": wearing,
                "selected": False,
                "motion": motion,
                "fallen": fallen,
            }
        )
        detections.append(
            {
                "label": f"person {track_id}" + (" FALLEN" if fallen else ""),
                "confidence": float(worker["score"]),
                "xyxy": worker["box"],
                "xyxy_norm": [
                    worker["box"][0] / width,
                    worker["box"][1] / height,
                    worker["box"][2] / width,
                    worker["box"][3] / height,
                ],
                "distance_m": distance_m,
                "distance_source": source,
                "distance_method": estimate.method if estimate else None,
                "distance_confidence": estimate.confidence if estimate else None,
                "vision_distance_m": estimate.body_m if estimate else None,
                "pose_used": bool(keypoints),
                "track_id": track_id,
                "approach_mps": motion["approach_mps"],
                "fallen": fallen,
                "fall_confidence": max((float(det["confidence"]) for det in fall_boxes), default=None),
                "keypoints": keypoints,
            }
        )
        for gear_box in wearing.get("detections") or []:
            drawn_gear.append({**gear_box, "track_id": track_id})

    if tracking:
        motion_log.prune({int(person["id"]) for person in people})
    person_count = len(people)
    worker_detected = person_count > 0
    selected = _select_person(people)
    if selected is not None:
        selected["selected"] = True
        for det in detections:
            det["is_primary"] = det.get("track_id") == selected["id"]
    if not worker_detected:
        label = "No person"
    elif person_count == 1:
        label = "Person"
    else:
        label = f"Person {selected['id']} · {person_count} in frame"
    if selected is not None:
        ppe_result = selected["ppe"]
    elif blocked is not None:
        ppe_result = blocked
    else:
        ppe_result = ppe.blank_wearing()
    ppe_result = dict(ppe_result)
    ppe_result["detections"] = drawn_gear
    annotated = _draw_detections(rgb, detections, drawn_gear + hazards["detections"]) if annotate else None
    motion = selected["motion"] if selected else {"speed_mps": None, "approach_mps": None, "approaching": None}
    match = next((det for det in detections if selected and det.get("track_id") == selected["id"]), None)
    distance_m = selected.get("distance_m") if selected else None
    source = selected.get("distance_source") if selected else None

    return {
        "label": label,
        "confidence": float(selected["confidence"]) if selected else None,
        "worker_detected": worker_detected,
        "person_count": person_count,
        "people": [
            {
                "id": person["id"],
                "selected": person["selected"],
                "summary": person["summary"],
                "distance_m": person["distance_m"],
                "distance_source": person["distance_source"],
                "approach_mps": person["motion"]["approach_mps"],
                "approaching": person["motion"]["approaching"],
                "fallen": person["fallen"],
                "confidence": person["confidence"],
                "ppe": person["ppe"],
            }
            for person in people
        ],
        "track_id": selected["id"] if selected else None,
        "tracked": tracking,
        "worker_speed": motion["speed_mps"],
        "approach_velocity": motion["approach_mps"],
        "approaching": motion["approaching"],
        "falls_available": bool(falls.get("available")),
        "fall_detected": any(person["fallen"] for person in people) or bool(lone_falls),
        "lone_falls": lone_falls,
        "frame_w": width,
        "frame_h": height,
        "hazards": {
            "available": hazards["available"],
            "model_id": hazards["model_id"],
            "fire_score": hazards["fire_score"],
            "smoke_score": hazards["smoke_score"],
            "detections": hazards["detections"],
        },
        "fire_score": hazards["fire_score"],
        "smoke_score": hazards["smoke_score"],
        "detections": detections,
        "worker_distance_est": distance_m,
        "distance_source": source,
        "vision_distance_m": match.get("vision_distance_m") if match else None,
        "distance_method": match.get("distance_method") if match else None,
        "distance_confidence": match.get("distance_confidence") if match else None,
        "tof_m": distance_m if source == "tof" else None,
        "distance_pairs": distance_pairs,
        "ppe": ppe_result,
        "hard_hat": ppe_result.get("hard_hat"),
        "vest": ppe_result.get("vest"),
        "gloves": ppe_result.get("gloves"),
        "boots": ppe_result.get("boots"),
        "ppe_complete": ppe_result.get("ppe_complete"),
        "ppe_summary": ppe_result.get("summary"),
        "unsafe": bool(ppe_result.get("ppe_complete") is False),
        "annotated_thumbnail": _image_to_data_url(annotated) if annotated is not None else None,
        "model_id": _model_id(),
        "backend": "ultralytics",
    }
