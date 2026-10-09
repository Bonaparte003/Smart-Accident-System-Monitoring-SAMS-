"""Scene behaviour classifier — safe and unsafe doings in the whole frame.

ResNet18 trained in ``notebooks/01_vision_safe_unsafe.ipynb`` on
Voxel51/Safe_and_Unsafe_Behaviours. It labels the whole frame, not one
person, and was trained on a single factory, so a label is reported only
after its probability, averaged over the last few calls, clears
``UNSAFE_MIN_PROB``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("monitoring.behaviour")

UNSAFE_CLASSES = {
    "Carrying Overload with Forklift",
    "Opened Panel Cover",
    "Safe Walkway Violation",
    "Unauthorized Intervention",
}
UNSAFE_MIN_PROB = 0.6
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_lock = threading.Lock()
_model = None
_classes: list[str] = []
_img_size = 224
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
    custom = os.environ.get("SAMS_BEHAVIOUR_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    return _project_root() / "models" / "vision" / "safe_unsafe" / "best.pt"


def metadata_path() -> Path:
    return weights_path().parent / "metadata.json"


def enabled() -> bool:
    return os.environ.get("SAMS_VISION_BEHAVIOUR", "true").strip().lower() in {"1", "true", "yes", "on"}


def get_status() -> dict[str, Any]:
    return {
        "enabled": enabled(),
        "available": enabled() and weights_path().is_file() and metadata_path().is_file() and _load_error is None,
        "weights": str(weights_path()),
        "error": _load_error,
    }


def get_model():
    global _model, _classes, _img_size, _load_error
    if not enabled() or not weights_path().is_file() or not metadata_path().is_file():
        return None
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    import torch
                    from torchvision import models

                    meta = json.loads(metadata_path().read_text())
                    checkpoint = torch.load(weights_path(), map_location="cpu", weights_only=False)
                    classes = list(checkpoint.get("class_names") or meta["class_names"])
                    net = models.resnet18(weights=None)
                    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
                    net.load_state_dict(checkpoint["model_state_dict"])
                    net.eval()
                    _classes = classes
                    _img_size = int(meta.get("img_size", 224))
                    _model = net
                    _load_error = None
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
                    logger.warning("behaviour model unavailable: %s", exc)
                    return None
    return _model


def classify(image) -> dict[str, Any] | None:
    """Class probabilities for one RGB frame, or None without the model."""
    model = get_model()
    if model is None:
        return None
    import torch

    rgb = image.convert("RGB").resize((_img_size, _img_size))
    array = (np.asarray(rgb, dtype=np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    return {name: float(probs[index]) for index, name in enumerate(_classes)}


class BehaviourWindow:
    """Average of the last few frame probabilities for one video or camera."""

    def __init__(self, size: int = 3) -> None:
        self._samples: deque[dict[str, float]] = deque(maxlen=size)

    def add(self, probs: dict[str, float] | None) -> dict[str, Any]:
        if probs:
            self._samples.append(probs)
        return self.current()

    def current(self) -> dict[str, Any]:
        if not self._samples:
            return {"available": False, "label": None, "prob": 0.0, "unsafe": False}
        names = list(self._samples[-1])
        mean = {name: sum(sample.get(name, 0.0) for sample in self._samples) / len(self._samples) for name in names}
        label = max(mean, key=mean.get)
        prob = round(mean[label], 3)
        return {
            "available": True,
            "label": label,
            "prob": prob,
            "unsafe": label in UNSAFE_CLASSES and prob >= UNSAFE_MIN_PROB,
        }
