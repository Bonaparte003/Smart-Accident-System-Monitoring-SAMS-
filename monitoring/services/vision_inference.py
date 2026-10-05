"""Vision worker detector — construction person boxes, pose, and measured distance.

Who is in the frame comes from the Construction-PPE person class. Pose runs
inside that box and only places joints. A pose that finds a limb and no head
or shoulders is rejected before it receives an id. A worker stays when pose
misses the hips, and when pose finds nothing at all. Gear counts only inside
that person's head, torso, hands, or feet. Distance is the time-of-flight
reading when the beam hits that torso, otherwise the trained distance
network on a fully visible body. Partial bodies publish no distance.

Live frames keep the track id the PPE checkpoint's tracker assigns. Each
track's distance history gives its approach velocity. A fall is the accident
model's box on that person. A separate fire/smoke detector runs on the whole frame.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from PIL import Image, ImageDraw, ImageFont

from . import distance_estimation as distance
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
    """Status payload used by the Vision lab model tag (kept name for API compat)."""
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
    try:
        from hardware.tof import get_tof

        return get_tof().read_metres()
    except Exception:
        return None


def _run_pose_frame(image: Image.Image, conf: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    pose = get_pose_model()
    if pose is None:
        return None
    try:
        cutoff = conf if conf is not None else (ppe.decision_conf() if ppe.spec_available() else 0.25)
        results = pose.predict(image, conf=cutoff, imgsz=int(_pose_spec.get("imgsz", 640)), verbose=False)
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


_motion = MotionLog()


def reset_tracks() -> None:
    """Drop live distance history when a camera session starts or stops."""
    _motion.reset()
    ppe.reset_tracker()
    hazard.reset()


def _crop_worker(image: Image.Image, box: list[float], pad_frac: float = 0.12):
    """Pad a worker box and crop it. Pose then sees that worker, not the whole frame."""
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x = max(1.0, (x2 - x1) * pad_frac)
    pad_y = max(1.0, (y2 - y1) * pad_frac)
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom)), (left, top)


def _joints_inside(image: Image.Image, box: list[float], conf: float | None = None) -> tuple[bool, list[list[float]] | None]:
    """Place pose joints inside one worker box.

    Returns (keep, keypoints). A pose that found only a limb is dropped.
    No pose result keeps the worker and leaves the joints empty.
    """
    cropped = _crop_worker(image, box)
    if cropped is None:
        return True, None
    crop, origin = cropped
    pack = _run_pose_frame(crop, conf)
    if pack is None:
        return True, None
    xy, kp_conf, _boxes, scores = pack
    ox, oy = origin
    chosen: tuple[float, list[list[float]]] | None = None
    for index in range(len(scores)):
        keypoints = _keypoints_for(xy, kp_conf, index)
        for point in keypoints:
            point[0] += float(ox)
            point[1] += float(oy)
        if not ppe.worker_has_head_or_shoulders(keypoints):
            continue
        score = float(scores[index])
        if chosen is None or score > chosen[0]:
            chosen = (score, keypoints)
    if chosen is None:
        return False, None
    return True, chosen[1]


def _keypoints_for(xy: np.ndarray, conf: np.ndarray, index: int) -> list[list[float]]:
    points = []
    for joint in range(xy.shape[1]):
        points.append([float(xy[index, joint, 0]), float(xy[index, joint, 1]), float(conf[index, joint])])
    return points


def _select_person(people: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The alert follows the person closest to the machine."""
    if not people:
        return None

    def rank(person: dict[str, Any]) -> tuple:
        distance_m = person.get("distance_m")
        return (distance_m if distance_m is not None else 1e9, -float(person["confidence"]))

    return min(people, key=rank)


def _image_to_data_url(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def predict_image(image: Image.Image, conf: float | None = None, *, live: bool = False, track: bool = False) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    keep_identity = live or track
    tof_m = _read_tof_m() if live else None
    workers, gear, blocked = ppe.detect_scene(rgb, conf=conf, track=keep_identity)
    observed: list[dict[str, Any]] = []
    if blocked is None:
        for worker in workers:
            keep, keypoints = _joints_inside(rgb, worker["box"], conf)
            if not keep:
                continue
            observed.append(
                {
                    "box": worker["box"],
                    "score": float(worker["score"]),
                    "keypoints": keypoints,
                    "track_id": worker.get("track_id"),
                }
            )
    else:
        gear = []

    track_ids: list[int] = []
    stable_ids: list[bool] = []
    for index, person in enumerate(observed):
        model_id = person.get("track_id")
        if keep_identity and model_id is not None:
            track_ids.append(int(model_id))
            stable_ids.append(True)
        else:
            track_ids.append(index + 1)
            stable_ids.append(False)
    hazards = hazard.detect_hazards(rgb)
    if keep_identity:
        hazards = hazard.smooth_scores(hazards)
    falls = hazard.detect_falls(rgb)

    aimed_index = None
    best_aim = -1.0
    for index, person in enumerate(observed):
        regions = ppe.body_regions(person["keypoints"], person["box"])
        person["regions"] = regions
        if distance.aim_hits(regions.get("torso"), (width, height)) and person["score"] > best_aim:
            best_aim = person["score"]
            aimed_index = index

    people: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    drawn_gear: list[dict[str, Any]] = []
    for index, person in enumerate(observed):
        regions = person["regions"]
        if blocked is not None:
            wearing = blocked
        else:
            wearing = ppe.assess_person(person["keypoints"], person["box"], gear, regions=regions)
        estimate = distance.measure_person(
            person["keypoints"],
            (width, height),
            tof_m=tof_m,
            sensor_aimed=index == aimed_index,
            box=person["box"],
            det_conf=person["score"],
        )
        distance_m = estimate.distance_m if estimate else None
        source = estimate.source if estimate else None
        track_id = int(track_ids[index])
        fall = hazard.fall_for_person(person["box"], falls)
        if stable_ids[index]:
            _motion.record_distance(track_id, distance_m)
            motion = _motion.motion(track_id)
        else:
            motion = {"speed_mps": 0.0, "approach_mps": 0.0, "approaching": False, "samples": 0}
        record = {
            "id": track_id,
            "confidence": person["score"],
            "xyxy": person["box"],
            "distance_m": distance_m,
            "distance_source": source,
            "summary": wearing.get("summary"),
            "ppe": wearing,
            "selected": False,
            "motion": motion,
            "fallen": fall["fallen"],
        }
        people.append(record)
        detections.append(
            {
                "label": f"person {track_id}" + (" FALLEN" if fall["fallen"] else ""),
                "confidence": person["score"],
                "xyxy": person["box"],
                "xyxy_norm": [
                    person["box"][0] / width,
                    person["box"][1] / height,
                    person["box"][2] / width,
                    person["box"][3] / height,
                ],
                "distance_m": distance_m,
                "distance_source": source,
                "distance_method": estimate.method if estimate else None,
                "distance_confidence": estimate.confidence if estimate else None,
                "z0_m": estimate.body_m if estimate else None,
                "vision_distance_m": estimate.body_m if estimate else None,
                "pose_used": bool(person["keypoints"]),
                "track_id": track_id,
                "approach_mps": motion["approach_mps"],
                "fallen": fall["fallen"],
                "lying_score": fall["lying_score"],
            }
        )
        for gear_box in wearing.get("detections") or []:
            drawn_gear.append({**gear_box, "track_id": track_id})

    person_count = len(people)
    worker_detected = person_count > 0
    if keep_identity and worker_detected:
        for person in people:
            person["ppe"] = ppe.stabilize_live_ppe(person["ppe"], track_id=str(person["id"]))
            person["summary"] = person["ppe"].get("summary")
        ppe.prune_live_votes({str(person["id"]) for person in people})
        _motion.prune({int(person["id"]) for person in people})
    elif keep_identity:
        ppe.prune_live_votes(set())
        _motion.prune(set())
    selected = _select_person(people)
    if selected is not None:
        selected["selected"] = True
        for det in detections:
            det["is_primary"] = det.get("track_id") == selected["id"]
    confidence = float(selected["confidence"]) if selected else 0.0
    if not worker_detected:
        label = "No person"
    elif person_count == 1:
        label = "Person"
    else:
        label = f"Person {selected['id']} · {person_count} in frame"
    if worker_detected and selected is not None:
        ppe_result = selected["ppe"]
    elif blocked is not None:
        ppe_result = blocked
    else:
        ppe_result = ppe.blank_wearing()
    ppe_result = dict(ppe_result)
    ppe_result["detections"] = drawn_gear
    annotated = _draw_detections(rgb, detections, drawn_gear + hazards["detections"])
    motion = selected["motion"] if selected else {"speed_mps": 0.0, "approach_mps": 0.0, "approaching": False}
    fall_detected = any(person["fallen"] for person in people)
    distance_m = selected.get("distance_m") if selected else None
    source = selected.get("distance_source") if selected else None
    body_m = None
    method = None
    dist_conf = None
    if selected:
        match = next((det for det in detections if det.get("track_id") == selected["id"]), None)
        if match:
            body_m = match.get("vision_distance_m")
            method = match.get("distance_method")
            dist_conf = match.get("distance_confidence")

    return {
        "label": label,
        "confidence": confidence,
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
                "fallen": person["fallen"],
            }
            for person in people
        ],
        "track_id": selected["id"] if selected else None,
        "tracked": keep_identity,
        "worker_speed": motion["speed_mps"],
        "approach_velocity": motion["approach_mps"],
        "approaching": bool(motion["approaching"]),
        "fall_detected": fall_detected,
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
        "vision_distance_m": body_m,
        "distance_method": method,
        "distance_confidence": dist_conf,
        "z0_m": body_m,
        "tof_m": distance_m if source == "tof" else None,
        "ppe": ppe_result,
        "hard_hat": ppe_result.get("hard_hat"),
        "vest": ppe_result.get("vest"),
        "gloves": ppe_result.get("gloves"),
        "boots": ppe_result.get("boots"),
        "ppe_complete": ppe_result.get("ppe_complete"),
        "ppe_summary": ppe_result.get("summary"),
        "unsafe": bool(ppe_result.get("ppe_complete") is False),
        "annotated_thumbnail": _image_to_data_url(annotated),
        "model_id": _model_id(),
        "backend": "ultralytics",
        "probabilities": {
            "person": confidence if worker_detected else 1.0 - confidence,
            "no_person": 0.0 if worker_detected else 1.0,
        },
    }


def predict_image_bytes(data: bytes, *, track: bool = False) -> dict[str, Any]:
    image = Image.open(io.BytesIO(data))
    return predict_image(image, track=track)


def predict_data_url(data_url: str) -> dict[str, Any]:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return predict_image_bytes(base64.b64decode(data_url))


def extract_video_frames(video_bytes: bytes, max_frames: int = 8) -> dict[str, Any]:
    """Sample evenly spaced frames and run person detection on each."""
    import cv2

    results: list[dict[str, Any]] = []
    fps = 0.0
    total = 0
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if total <= 0:
            frames_bgr = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame)
            total = len(frames_bgr)
            indices = (
                np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
                if total
                else []
            )
            selected = [frames_bgr[i] for i in indices]
        else:
            indices = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
            selected = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    selected.append(frame)
        cap.release()

    for i, frame in enumerate(selected):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        pred = None
        error = None
        thumb = None
        try:
            pred = predict_image(image)
            thumb = pred.get("annotated_thumbnail")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            thumb = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        results.append(
            {
                "index": i,
                "thumbnail": thumb,
                "prediction": pred,
                "error": error,
            }
        )
    return {
        "frame_count_sampled": len(results),
        "fps": fps,
        "total_frames": total,
        "frames": results,
        "torch": get_torch_status(),
    }
