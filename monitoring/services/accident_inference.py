"""Fallen-worker detector.

The class names, confidence, and image size come from
``models/vision/accident/metadata.json``, written by
``notebooks/07_vision_accident_detection_training.ipynb``. A person is fallen
only when that model places a fall-class box on them. Without the weights,
the result is unavailable and nobody is marked fallen.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("monitoring.accident")

_lock = threading.Lock()
_model = None
_load_error: str | None = None


def _project_root() -> Path:
    try:
        from django.conf import settings

        if settings.configured:
            return Path(settings.BASE_DIR)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[2]


def weights_path() -> Path:
    custom = os.environ.get("SAMS_ACCIDENT_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    return _project_root() / "models" / "vision" / "accident" / "best.pt"


def metadata_path() -> Path:
    return weights_path().parent / "metadata.json"


def load_spec() -> dict[str, Any]:
    path = metadata_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Accident metadata not found at {path}. "
            "Export it with notebooks/07_vision_accident_detection_training.ipynb"
        )
    spec = json.loads(path.read_text())
    for key in ("fall_classes", "conf", "imgsz", "model_id"):
        if key not in spec:
            raise ValueError(f"Accident metadata at {path} is missing {key}")
    return spec


def model_id() -> str:
    try:
        return str(load_spec()["model_id"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "accident"


def available() -> bool:
    return weights_path().is_file() and metadata_path().is_file() and _load_error is None


def get_status() -> dict[str, Any]:
    return {
        "available": available(),
        "weights": str(weights_path()) if weights_path().is_file() else None,
        "model_id": model_id(),
        "error": _load_error,
    }


def get_model():
    global _model, _load_error
    if not weights_path().is_file():
        return None
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from ultralytics import YOLO

                    _model = YOLO(str(weights_path()))
                    _load_error = None
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
                    logger.warning("accident model unavailable: %s", exc)
                    return None
    return _model


def _norm(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("-", " ").replace("_", " ").split())


def detect_falls(image) -> dict[str, Any]:
    """Fall-class boxes from the accident checkpoint, or an unavailable result."""
    result: dict[str, Any] = {
        "available": False,
        "model_id": model_id(),
        "detections": [],
    }
    if not metadata_path().is_file() or not weights_path().is_file():
        return result
    try:
        spec = load_spec()
        model = get_model()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("accident metadata unusable: %s", exc)
        return result
    if model is None:
        return result
    result["available"] = True
    result["model_id"] = spec["model_id"]
    fall_names = {_norm(name) for name in spec["fall_classes"]}
    try:
        outputs = model.predict(image, conf=float(spec["conf"]), imgsz=int(spec["imgsz"]), verbose=False)
    except Exception as exc:
        logger.warning("accident predict failed: %s", exc)
        result["available"] = False
        return result
    if not outputs:
        return result
    boxes = getattr(outputs[0], "boxes", None)
    names = getattr(outputs[0], "names", None) or getattr(model, "names", {}) or {}
    if boxes is None or not len(boxes):
        return result
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for index in range(len(xyxy)):
        label = str(names.get(int(classes[index]), classes[index]))
        if _norm(label) not in fall_names:
            continue
        result["detections"].append(
            {
                "label": label,
                "confidence": float(scores[index]),
                "xyxy": [float(v) for v in xyxy[index]],
            }
        )
    return result


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def fall_for_box(box: list[float] | None, falls: dict[str, Any]) -> dict[str, Any]:
    """The fall detection that overlaps this person, if the model found one."""
    if not falls.get("available") or not box:
        return {"lying_score": 0.0, "fallen": False}
    best = 0.0
    for det in falls.get("detections") or []:
        if _iou(box, det["xyxy"]) <= 0:
            continue
        best = max(best, float(det["confidence"]))
    return {"lying_score": round(best, 3), "fallen": best > 0}
