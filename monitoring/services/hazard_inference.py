"""Scene hazards: fire and smoke boxes, and a fallen worker.

Fire and smoke come from the detector trained in
``notebooks/06_vision_hazard_detection_training.ipynb``. Which output name is
fire or smoke is that notebook's ``metadata.json``. A fall comes from the
accident detector (``accident_inference``), not from the pose angle.

Without those weights the detectors report ``available=False`` and zero scores.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any

from . import accident_inference as accident

logger = logging.getLogger("monitoring.hazard")

_lock = threading.Lock()
_model = None
_load_error: str | None = None
_scores: dict[str, deque[float]] = {"fire": deque(), "smoke": deque()}


def _project_root() -> Path:
    try:
        from django.conf import settings

        if settings.configured:
            return Path(settings.BASE_DIR)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[2]


def metadata_path() -> Path:
    return weights_path().parent / "metadata.json"


def load_spec() -> dict[str, Any]:
    path = metadata_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Hazard metadata not found at {path}. "
            "Export it with notebooks/06_vision_hazard_detection_training.ipynb"
        )
    spec = json.loads(path.read_text())
    for key in ("classes", "conf", "imgsz", "model_id", "score_window"):
        if key not in spec:
            raise ValueError(f"Hazard metadata at {path} is missing {key}")
    return spec


def model_id() -> str:
    try:
        return str(load_spec()["model_id"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "hazard"


def weights_path() -> Path:
    custom = os.environ.get("SAMS_HAZARD_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    return _project_root() / "models" / "vision" / "hazard" / "best.pt"


def hazard_enabled() -> bool:
    return os.environ.get("SAMS_VISION_HAZARD", "true").strip().lower() in {"1", "true", "yes", "on"}


def get_status() -> dict[str, Any]:
    path = weights_path()
    return {
        "enabled": hazard_enabled(),
        "weights": str(path),
        "available": hazard_enabled() and path.is_file() and metadata_path().is_file(),
        "model_id": model_id(),
        "accident": accident.get_status(),
        "error": _load_error,
    }


def get_model():
    global _model, _load_error
    if not hazard_enabled() or not weights_path().is_file():
        return None
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from ultralytics import YOLO

                    _model = YOLO(str(weights_path()))
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
                    logger.warning("fire/smoke model unavailable: %s", exc)
                    return None
    return _model


def _norm(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def _kind(label: str, classes: dict[str, Any]) -> str | None:
    norm = _norm(label)
    for kind, names in classes.items():
        if norm in {_norm(name) for name in names or []}:
            return kind
    return None


def detect_hazards(image, *, conf: float | None = None) -> dict[str, Any]:
    """Fire / smoke boxes in one frame and the strongest score of each."""
    result: dict[str, Any] = {
        "available": False,
        "model_id": model_id(),
        "fire_score": 0.0,
        "smoke_score": 0.0,
        "detections": [],
    }
    if not metadata_path().is_file():
        return result
    try:
        spec = load_spec()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("hazard metadata unusable: %s", exc)
        return result
    model = get_model()
    if model is None:
        return result
    result["available"] = True
    result["model_id"] = spec["model_id"]
    cutoff = float(spec["conf"] if conf is None else conf)
    try:
        outputs = model.predict(image, conf=cutoff, imgsz=int(spec["imgsz"]), verbose=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fire/smoke predict failed: %s", exc)
        result["available"] = False
        return result
    class_map = spec["classes"]
    if not outputs:
        return result
    boxes = getattr(outputs[0], "boxes", None)
    names = getattr(outputs[0], "names", None) or getattr(model, "names", {}) or {}
    if boxes is None or not len(boxes):
        return result
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    for index in range(len(xyxy)):
        label = str(names.get(int(class_ids[index]), class_ids[index]))
        kind = _kind(label, class_map)
        if kind is None:
            continue
        score = float(scores[index])
        result[f"{kind}_score"] = max(result[f"{kind}_score"], score)
        result["detections"].append(
            {
                "label": kind,
                "confidence": score,
                "xyxy": [float(v) for v in xyxy[index]],
                "missing": True,
            }
        )
    return result


def smooth_scores(result: dict[str, Any]) -> dict[str, Any]:
    """Mean of the last few live frames, using the window stored with the hazard model."""
    if not result.get("available"):
        return result
    try:
        window = max(1, int(load_spec()["score_window"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return result
    out = dict(result)
    for kind in ("fire", "smoke"):
        samples = _scores[kind]
        samples.append(float(result.get(f"{kind}_score") or 0.0))
        while len(samples) > window:
            samples.popleft()
        out[f"{kind}_score"] = round(sum(samples) / len(samples), 4)
    return out


def detect_falls(image) -> dict[str, Any]:
    return accident.detect_falls(image)


def fall_for_person(box: list[float] | None, falls: dict[str, Any]) -> dict[str, Any]:
    return accident.fall_for_box(box, falls)


def reset() -> None:
    for queue in _scores.values():
        queue.clear()
