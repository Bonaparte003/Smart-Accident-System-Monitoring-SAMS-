"""Lazy-load and run the selected fusion risk classifier.

The model's own metadata names its input columns and the zone limits its
training data used, so a retrained model with more inputs drops in without
code changes. Every column must be supplied; nothing is filled in here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings

_lock = threading.Lock()
_model = None
_metadata: dict[str, Any] | None = None


def _model_dir() -> Path:
    return Path(settings.BASE_DIR) / "models" / "fusion" / "best_fusion_model"


def get_metadata() -> dict[str, Any]:
    global _metadata
    if _metadata is None:
        meta_path = _model_dir() / "metadata.json"
        with meta_path.open() as f:
            _metadata = json.load(f)
    return _metadata


def get_feature_cols() -> list[str]:
    return list(get_metadata()["feature_cols"])


def class_names() -> list[str]:
    """Risk levels in the model's class order."""
    return list(get_metadata()["class_names"])


def motion_window_s() -> float:
    """Seconds of distance history that speed and approach were measured over for this model."""
    return float(get_metadata()["motion_window_s"])


def zones_m() -> dict[str, float]:
    """Caution and danger limits the fusion training data was labelled with."""
    zones = get_metadata()["zones_m"]
    return {"caution": float(zones["caution"]), "danger": float(zones["danger"])}


def absent_worker_values() -> dict[str, float]:
    """The person inputs the model was trained with for a frame with nobody in it.

    Empty for a model trained without such rows: there, a frame with nobody in
    it leaves the person inputs unmeasured.
    """
    return {name: float(value) for name, value in (get_metadata().get("absent_worker") or {}).items()}


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = joblib.load(_model_dir() / "model.joblib")
    return _model


def reload_model() -> None:
    global _model, _metadata
    with _lock:
        _model = None
        _metadata = None


def predict_fusion(features: dict[str, float]) -> dict[str, Any]:
    """Run the fusion model on a feature dict that holds every column it was trained on."""
    cols = get_feature_cols()
    missing = [c for c in cols if c not in features]
    if missing:
        raise KeyError(f"Fusion inputs missing: {missing}")
    row = np.array([[float(features[c]) for c in cols]], dtype=np.float32)
    model = get_model()
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    names = class_names()
    probabilities = {name: 0.0 for name in names}
    for index, cls in enumerate(classes):
        probabilities[names[cls]] = float(proba[index])
    label = max(names, key=lambda name: probabilities[name])
    return {
        "label": label,
        "label_idx": names.index(label),
        "confidence": float(probabilities[label]),
        "probabilities": probabilities,
        "model_id": get_metadata().get("experiment_id", "best_fusion_model"),
        "feature_cols": cols,
    }
