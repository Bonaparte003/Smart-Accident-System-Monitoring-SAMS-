"""Lazy-load and run the selected fusion risk classifier.

The model's own metadata names its input columns, so a retrained model with
more inputs (hazards, approach velocity) drops in without code changes. A
feature the caller does not supply is sent as 0.0.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings

BASE_FEATURE_COLS = [
    "worker_detected",
    "worker_distance",
    "inside_caution_zone",
    "inside_danger_zone",
    "worker_speed",
    "approaching_machine",
    "vision_confidence",
    "machine_active",
    "vibration_state",
    "vibration_anomaly_score",
]
HAZARD_FEATURE_COLS = [
    "approach_velocity",
    "fire_score",
    "smoke_score",
    "fall_detected",
]
FEATURE_COLS = BASE_FEATURE_COLS

CLASS_NAMES = ["NORMAL", "CAUTION", "DANGER"]
CAUTION_ZONE_M = 2.0
DANGER_ZONE_M = 0.8

DEFAULT_DECISION = {
    "danger_threshold": 0.35,
    "caution_threshold": 0.45,
    "instant_danger": 0.7,
    "smoothing_windows": 3,
}

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
    cols = get_metadata().get("feature_cols")
    return list(cols) if cols else list(BASE_FEATURE_COLS)


def get_decision_config() -> dict[str, float]:
    return {**DEFAULT_DECISION, **(get_metadata().get("decision") or {})}


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
    """Run the fusion model on a feature dict. Missing columns are 0.0."""
    cols = get_feature_cols()
    row = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=np.float32)
    model = get_model()
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    probabilities = {name: 0.0 for name in CLASS_NAMES}
    for index, cls in enumerate(classes):
        probabilities[CLASS_NAMES[cls]] = float(proba[index])
    label = max(CLASS_NAMES, key=lambda name: probabilities[name])
    return {
        "label": label,
        "label_idx": CLASS_NAMES.index(label),
        "confidence": float(probabilities[label]),
        "probabilities": probabilities,
        "model_id": get_metadata().get("experiment_id", "best_fusion_model"),
        "feature_cols": cols,
    }
