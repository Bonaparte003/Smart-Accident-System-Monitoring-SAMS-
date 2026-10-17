"""Vibration state for one accelerometer window.

The label and its probabilities are the classifier trained in
``notebooks/02_vibration_uci_accelerometer.ipynb``, as it outputs them. Its
metadata names the window length, sample rate, feature columns and classes.
Without that model there is no vibration state.
"""

from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings
from scipy.stats import kurtosis, skew

_lock = threading.Lock()
_model = None
_metadata: dict[str, Any] | None = None
_load_error: str | None = None


def model_dir() -> Path:
    return Path(settings.BASE_DIR) / "models" / "vibration" / "best"


def get_model():
    """The promoted classifier, or None when it is missing or cannot give probabilities."""
    global _model, _metadata, _load_error
    if _model is None and _load_error is None:
        with _lock:
            if _model is None and _load_error is None:
                folder = model_dir()
                try:
                    model = joblib.load(folder / "model.joblib")
                    if not hasattr(model, "predict_proba"):
                        raise TypeError(
                            f"{folder / 'model.joblib'} has no predict_proba. "
                            "Retrain with notebooks/02_vibration_uci_accelerometer.ipynb"
                        )
                    _metadata = json.loads((folder / "metadata.json").read_text())
                    _model = model
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
    return _model


def get_metadata() -> dict[str, Any]:
    get_model()
    return _metadata or {}


def get_status() -> dict[str, Any]:
    model = get_model()
    return {
        "available": model is not None,
        "experiment_id": get_metadata().get("experiment_id"),
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


def _axis_features(sig: np.ndarray, prefix: str, sample_rate_hz: float) -> dict[str, float]:
    sig = sig.astype(np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    n = len(sig)
    fft = np.fft.rfft(sig - sig.mean())
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mag = np.abs(fft)
    if len(mag) > 1:
        dom_idx = 1 + int(np.argmax(mag[1:]))
        dom_freq = float(freqs[dom_idx])
        dom_amp = float(mag[dom_idx])
    else:
        dom_freq, dom_amp = 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kurt = float(kurtosis(sig, fisher=True, bias=False))
        skw = float(skew(sig, bias=False))
    return {
        f"{prefix}_mean": float(sig.mean()),
        f"{prefix}_std": float(sig.std()),
        f"{prefix}_rms": rms,
        f"{prefix}_max": float(sig.max()),
        f"{prefix}_min": float(sig.min()),
        f"{prefix}_ptp": float(np.ptp(sig)),
        f"{prefix}_energy": float(np.sum(sig ** 2)),
        f"{prefix}_var": float(sig.var()),
        f"{prefix}_kurtosis": kurt if np.isfinite(kurt) else 0.0,
        f"{prefix}_skew": skw if np.isfinite(skw) else 0.0,
        f"{prefix}_dom_freq": dom_freq,
        f"{prefix}_dom_amp": dom_amp,
    }


def window_to_features(win: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    """12 statistics per axis and for |a|. Kurtosis and skew of a flat signal are 0, not NaN."""
    mag = np.sqrt(np.sum(win.astype(np.float64) ** 2, axis=1))
    feats: dict[str, float] = {}
    for i, axis in enumerate(["x", "y", "z"]):
        feats.update(_axis_features(win[:, i], axis, sample_rate_hz))
    feats.update(_axis_features(mag, "mag", sample_rate_hz))
    return feats


def _series(win: np.ndarray) -> dict[str, list[float]]:
    return {"x": win[:, 0].tolist(), "y": win[:, 1].tolist(), "z": win[:, 2].tolist()}


def _snapshot(feats: dict[str, float]) -> dict[str, float]:
    return {k: round(feats[k], 4) for k in ("x_rms", "y_rms", "z_rms", "mag_rms", "mag_std")}


def feature_cols() -> list[str]:
    return list(get_metadata()["feature_cols"])


def predict_from_window(win: np.ndarray | None) -> dict[str, Any] | None:
    """The trained classifier's label for one window of the model's length, or None without one."""
    model = get_model()
    if model is None or win is None or len(win) != window_size():
        return None
    meta = get_metadata()
    names = list(meta["class_names"])
    feats = window_to_features(win, float(meta["sample_rate_hz"]))
    row = np.array([[feats[c] for c in feature_cols()]], dtype=np.float32)
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]
    probabilities = {name: 0.0 for name in names}
    for index, cls in enumerate(classes):
        probabilities[names[cls]] = float(proba[index])
    label = max(names, key=lambda name: probabilities[name])
    return {
        "label": label,
        "label_idx": names.index(label),
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "abnormal": label == "ABNORMAL",
        "anomaly_score": round(probabilities["ABNORMAL"], 4),
        "model_id": get_metadata().get("experiment_id", "vibration"),
        "series": _series(win),
        "feature_snapshot": _snapshot(feats),
        "mag_std": round(float(feats["mag_std"]), 4),
    }
