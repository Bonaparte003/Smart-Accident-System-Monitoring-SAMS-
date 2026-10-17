"""Machine state for one accelerometer window: STOPPED or RUNNING.

The classifier is trained in ``notebooks/02_vibration_uci_accelerometer.ipynb``
(section 8) from phone recordings of a machine at rest and the UCI fan
running. Its label is used as it comes out. Without the model, machine state
is not measured.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings

from .vibration_inference import window_to_features

_lock = threading.Lock()
_model = None
_metadata: dict[str, Any] | None = None
_load_error: str | None = None


def model_dir() -> Path:
    return Path(settings.BASE_DIR) / "models" / "machine_state"


def get_model():
    """The trained classifier, or None when it is missing."""
    global _model, _metadata, _load_error
    if _model is None and _load_error is None:
        with _lock:
            if _model is None and _load_error is None:
                try:
                    _metadata = json.loads((model_dir() / "metadata.json").read_text())
                    _model = joblib.load(model_dir() / "model.joblib")
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
    return _model


def get_metadata() -> dict[str, Any]:
    get_model()
    return _metadata or {}


def get_status() -> dict[str, Any]:
    return {
        "available": get_model() is not None,
        "model_id": get_metadata().get("model_id"),
        "error": _load_error,
    }


def reload_model() -> None:
    global _model, _metadata, _load_error
    with _lock:
        _model = None
        _metadata = None
        _load_error = None


def window_size() -> int | None:
    """Samples per window the model was trained on, or None without the model."""
    return int(get_metadata()["window_size"]) if get_model() is not None else None


def predict_window(win: np.ndarray | None) -> dict[str, Any] | None:
    """The classifier's machine state for one window of the model's length, or None without one."""
    model = get_model()
    if model is None or win is None or len(win) != window_size():
        return None
    meta = get_metadata()
    names = list(meta["class_names"])
    feats = window_to_features(win, float(meta["sample_rate_hz"]))
    row = np.array([[feats[c] for c in meta["feature_cols"]]], dtype=np.float32)
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    probabilities = {name: 0.0 for name in names}
    for index, cls in enumerate(classes):
        probabilities[names[cls]] = float(proba[index])
    label = max(names, key=lambda name: probabilities[name])
    return {
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "running": label == "RUNNING",
        "model_id": meta.get("model_id", "machine_state"),
    }
