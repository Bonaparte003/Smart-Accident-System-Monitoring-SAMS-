"""Lazy-load and run the selected fusion risk classifier."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings

FEATURE_COLS = [
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

CLASS_NAMES = ["NORMAL", "CAUTION", "DANGER"]
CAUTION_ZONE_M = 2.0
DANGER_ZONE_M = 0.8

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


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = joblib.load(_model_dir() / "model.joblib")
    return _model


def predict_fusion(features: dict[str, float]) -> dict[str, Any]:
    """Run fusion model on a feature dict matching FEATURE_COLS."""
    row = np.array([[float(features[c]) for c in FEATURE_COLS]], dtype=np.float32)
    model = get_model()
    pred_idx = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}
    label = CLASS_NAMES[pred_idx]
    return {
        "label": label,
        "label_idx": pred_idx,
        "confidence": float(probabilities[label]),
        "probabilities": probabilities,
        "model_id": get_metadata().get("experiment_id", "best_fusion_model"),
    }
