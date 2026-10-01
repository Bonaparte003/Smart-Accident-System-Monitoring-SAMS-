"""Vision worker detector — YOLOv8 person detection + residual distance NN.

Person boxes come from ultralytics YOLOv8n. Distance uses a pose-geometry inspired
residual MLP (Z = Z0 * exp(δ)) for far range, fused with optional ToF for near range.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings
from PIL import Image, ImageDraw, ImageFont

from . import distance_estimation as distance
from . import ppe_inference as ppe

# COCO class id for person
PERSON_CLASS_ID = 0
MODEL_ID = "yolov8n_person+distance_nn"
DEFAULT_CONF = 0.35

_lock = threading.Lock()
_model = None
_pose_model = None
_yolo_error: str | None = None


def _model_weights_path() -> Path:
    return Path(settings.BASE_DIR) / "models" / "vision" / "yolov8n.pt"


def _pose_weights_path() -> Path:
    return Path(settings.BASE_DIR) / "models" / "vision" / "yolov8n-pose.pt"


def _pose_enabled() -> bool:
    # Off by default — enable with SAMS_VISION_POSE=1 (downloads yolov8n-pose on first use).
    return os.environ.get("SAMS_VISION_POSE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        global _yolo_error
        _yolo_error = str(exc)
        return False


def get_torch_status() -> dict[str, Any]:
    """Status payload used by the Vision lab model tag (kept name for API compat)."""
    ok = yolo_available()
    calib = distance.load_calib()
    return {
        "available": ok,
        "model_id": MODEL_ID,
        "detail": None if ok else (_yolo_error or "ultralytics is not installed."),
        "task": "person_detection+distance+ppe",
        "class_names": ["person"],
        "distance": {
            "method": "residual_mlp_pinhole",
            "weights": distance.weights_path().is_file(),
            "pose_enabled": _pose_enabled(),
            "tof_near_m": calib.get("tof_near_m"),
            "tof_far_m": calib.get("tof_far_m"),
            "fy_px": calib.get("fy_px"),
        },
        "ppe": ppe.get_status(),
    }


def get_model():
    global _model
    if not yolo_available():
        raise RuntimeError(_yolo_error or "ultralytics unavailable")
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO

                weights = _model_weights_path()
                if weights.is_file():
                    model = YOLO(str(weights))
                else:
                    # First run downloads COCO yolov8n into models/vision/
                    weights.parent.mkdir(parents=True, exist_ok=True)
                    model = YOLO("yolov8n.pt")
                    try:
                        src = Path(getattr(model, "ckpt_path", None) or "yolov8n.pt")
                        if src.is_file() and src.resolve() != weights.resolve():
                            weights.write_bytes(src.read_bytes())
                    except OSError:
                        pass
                _model = model
    return _model


def get_pose_model():
    """Optional YOLOv8n-pose for keypoint-enhanced distance features."""
    global _pose_model
    if not _pose_enabled() or not yolo_available():
        return None
    if _pose_model is None:
        with _lock:
            if _pose_model is None:
                try:
                    from ultralytics import YOLO

                    weights = _pose_weights_path()
                    if weights.is_file():
                        _pose_model = YOLO(str(weights))
                    else:
                        weights.parent.mkdir(parents=True, exist_ok=True)
                        _pose_model = YOLO("yolov8n-pose.pt")
                        try:
                            src = Path(getattr(_pose_model, "ckpt_path", None) or "yolov8n-pose.pt")
                            if src.is_file() and src.resolve() != weights.resolve():
                                weights.write_bytes(src.read_bytes())
                        except OSError:
                            pass
                except Exception as exc:  # noqa: BLE001
                    import logging

                    logging.getLogger("monitoring.vision").warning("pose model unavailable: %s", exc)
                    return None
    return _pose_model


def _read_tof_m() -> float | None:
    try:
        from hardware.tof import get_tof

        return get_tof().read_metres()
    except Exception:  # noqa: BLE001
        return None


def _match_pose_keypoints(
    pose_xy: np.ndarray,
    pose_conf: np.ndarray,
    pose_boxes: np.ndarray,
    box_xyxy: list[float],
) -> list[list[float]] | None:
    bx1, by1, bx2, by2 = box_xyxy
    best_i = None
    best_iou = 0.0
    for i, pb in enumerate(pose_boxes):
        ix1 = max(bx1, float(pb[0]))
        iy1 = max(by1, float(pb[1]))
        ix2 = min(bx2, float(pb[2]))
        iy2 = min(by2, float(pb[3]))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (bx2 - bx1) * (by2 - by1) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
        iou = inter / max(union, 1e-6)
        if iou > best_iou:
            best_iou = iou
            best_i = i
    if best_i is None or best_iou < 0.2:
        return None
    pts = []
    for j in range(pose_xy.shape[1]):
        pts.append(
            [
                float(pose_xy[best_i, j, 0]),
                float(pose_xy[best_i, j, 1]),
                float(pose_conf[best_i, j]),
            ]
        )
    return pts


def _run_pose_frame(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pose = get_pose_model()
    if pose is None:
        return None
    try:
        results = pose.predict(image, conf=0.3, verbose=False)
    except Exception:  # noqa: BLE001
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
    return xy, conf, bboxes


def _draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    ppe_detections: list[dict[str, Any]] | None = None,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
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
        missing = name.startswith("no_")
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


def _image_to_data_url(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def predict_image(image: Image.Image, conf: float = DEFAULT_CONF) -> dict[str, Any]:
    model = get_model()
    rgb = image.convert("RGB")
    w, h = rgb.size
    tof_m = _read_tof_m()

    results = model.predict(rgb, conf=conf, classes=[PERSON_CLASS_ID], verbose=False)
    pose_pack = _run_pose_frame(rgb)
    detections: list[dict[str, Any]] = []
    if results:
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                box = [float(v) for v in xyxy[i]]
                score = float(scores[i])
                kps = None
                if pose_pack is not None:
                    kps = _match_pose_keypoints(pose_pack[0], pose_pack[1], pose_pack[2], box)
                try:
                    est = distance.estimate_person_distance(
                        box,
                        (w, h),
                        det_conf=score,
                        keypoints=kps,
                        tof_m=tof_m,
                    )
                except Exception:  # noqa: BLE001
                    # Absolute fallback to legacy heuristic
                    d = distance.heuristic_distance_m(box, h)
                    est = distance.DistanceEstimate(
                        distance_m=d,
                        z0_m=d,
                        vision_m=d,
                        source="heuristic",
                        confidence=0.4,
                        method="legacy_bbox_height",
                    )
                detections.append(
                    {
                        "label": "person",
                        "confidence": score,
                        "xyxy": box,
                        "xyxy_norm": [box[0] / w, box[1] / h, box[2] / w, box[3] / h],
                        "distance_m": est.distance_m,
                        "distance_source": est.source,
                        "distance_method": est.method,
                        "distance_confidence": est.confidence,
                        "z0_m": est.z0_m,
                        "vision_distance_m": est.vision_m,
                        "pose_used": bool(kps),
                    }
                )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    if detections:
        detections[0]["is_primary"] = True

    person_count = len(detections)
    worker_detected = person_count > 0
    primary = detections[0] if detections else None
    confidence = float(primary["confidence"]) if primary else 0.0
    distance_m = float(primary["distance_m"]) if primary else None
    label = f"Person ×{person_count}" if worker_detected else "No person"

    ppe_result = ppe.predict_ppe(rgb, conf=conf, person_hint=worker_detected)
    annotated = _draw_detections(rgb, detections, ppe_result.get("detections"))

    payload: dict[str, Any] = {
        "label": label,
        "confidence": confidence,
        "worker_detected": worker_detected,
        "person_count": person_count,
        "detections": detections,
        "worker_distance_est": distance_m,
        "distance_source": primary.get("distance_source") if primary else None,
        "distance_method": primary.get("distance_method") if primary else None,
        "distance_confidence": primary.get("distance_confidence") if primary else None,
        "z0_m": primary.get("z0_m") if primary else None,
        "tof_m": tof_m,
        "ppe": ppe_result,
        "hard_hat": ppe_result.get("hard_hat"),
        "vest": ppe_result.get("vest"),
        "goggles": ppe_result.get("goggles"),
        "gloves": ppe_result.get("gloves"),
        "ppe_complete": ppe_result.get("ppe_complete"),
        "ppe_summary": ppe_result.get("summary"),
        "unsafe": bool(ppe_result.get("ppe_complete") is False),
        "annotated_thumbnail": _image_to_data_url(annotated),
        "model_id": MODEL_ID,
        "backend": "ultralytics",
        "probabilities": {
            "person": confidence if worker_detected else 1.0 - confidence,
            "no_person": 0.0 if worker_detected else 1.0,
        },
    }
    return payload


def predict_image_bytes(data: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(data))
    return predict_image(image)


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
