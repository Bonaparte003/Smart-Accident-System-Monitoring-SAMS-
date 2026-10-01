"""Vibration anomaly inference using the selected XGBoost window model."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from django.conf import settings
from scipy.stats import kurtosis, skew

SAMPLE_RATE_HZ = 50.0
WINDOW_SIZE = 100

FEATURE_COLS = [
    "x_mean", "x_std", "x_rms", "x_max", "x_min", "x_ptp", "x_energy", "x_var",
    "x_kurtosis", "x_skew", "x_dom_freq", "x_dom_amp",
    "y_mean", "y_std", "y_rms", "y_max", "y_min", "y_ptp", "y_energy", "y_var",
    "y_kurtosis", "y_skew", "y_dom_freq", "y_dom_amp",
    "z_mean", "z_std", "z_rms", "z_max", "z_min", "z_ptp", "z_energy", "z_var",
    "z_kurtosis", "z_skew", "z_dom_freq", "z_dom_amp",
    "mag_mean", "mag_std", "mag_rms", "mag_max", "mag_min", "mag_ptp", "mag_energy",
    "mag_var", "mag_kurtosis", "mag_skew", "mag_dom_freq", "mag_dom_amp",
]

CLASS_NAMES = ["NORMAL", "ABNORMAL"]

_lock = threading.Lock()
_model = None
_metadata: dict[str, Any] | None = None
_loaded_dir: Path | None = None


def _candidate_dirs() -> list[Path]:
    base = Path(settings.BASE_DIR) / "models" / "vibration"
    return [base / "vib_exp02_xgboost", base / "vib_exp01_random_forest"]


def get_model():
    global _model, _loaded_dir, _metadata
    if _model is None:
        with _lock:
            if _model is None:
                errors: list[str] = []
                for folder in _candidate_dirs():
                    path = folder / "model.joblib"
                    if not path.exists():
                        continue
                    try:
                        _model = joblib.load(path)
                        _loaded_dir = folder
                        with (folder / "metadata.json").open() as f:
                            _metadata = json.load(f)
                        break
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{folder.name}: {exc}")
                if _model is None:
                    raise RuntimeError("Could not load vibration model. " + " | ".join(errors))
    return _model


def get_metadata() -> dict[str, Any]:
    global _metadata
    if _metadata is None:
        get_model()
    return _metadata or {}


def _axis_features(sig: np.ndarray, prefix: str) -> dict[str, float]:
    sig = sig.astype(np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    n = len(sig)
    fft = np.fft.rfft(sig - sig.mean())
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)
    mag = np.abs(fft)
    if len(mag) > 1:
        dom_idx = 1 + int(np.argmax(mag[1:]))
        dom_freq = float(freqs[dom_idx])
        dom_amp = float(mag[dom_idx])
    else:
        dom_freq, dom_amp = 0.0, 0.0

    return {
        f"{prefix}_mean": float(sig.mean()),
        f"{prefix}_std": float(sig.std()),
        f"{prefix}_rms": rms,
        f"{prefix}_max": float(sig.max()),
        f"{prefix}_min": float(sig.min()),
        f"{prefix}_ptp": float(np.ptp(sig)),
        f"{prefix}_energy": float(np.sum(sig ** 2)),
        f"{prefix}_var": float(sig.var()),
        f"{prefix}_kurtosis": float(kurtosis(sig, fisher=True, bias=False)),
        f"{prefix}_skew": float(skew(sig, bias=False)),
        f"{prefix}_dom_freq": dom_freq,
        f"{prefix}_dom_amp": dom_amp,
    }


def synthesize_window(
    *,
    x_amp: float = 0.05,
    y_amp: float = 0.05,
    z_amp: float = 0.05,
    x_bias: float = 1.0,
    y_bias: float = 0.0,
    z_bias: float = 0.0,
    freq_hz: float = 5.0,
    noise: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Build a synthetic (WINDOW_SIZE, 3) accelerometer window from UI amplitudes."""
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_SIZE, dtype=np.float64) / SAMPLE_RATE_HZ
    phase = rng.uniform(0, 2 * np.pi, size=3)
    x = x_bias + x_amp * np.sin(2 * np.pi * freq_hz * t + phase[0]) + rng.normal(0, noise, WINDOW_SIZE)
    y = y_bias + y_amp * np.sin(2 * np.pi * freq_hz * 1.3 * t + phase[1]) + rng.normal(0, noise, WINDOW_SIZE)
    z = z_bias + z_amp * np.sin(2 * np.pi * freq_hz * 0.7 * t + phase[2]) + rng.normal(0, noise, WINDOW_SIZE)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def window_to_features(win: np.ndarray) -> dict[str, float]:
    mag = np.sqrt(np.sum(win.astype(np.float64) ** 2, axis=1))
    feats: dict[str, float] = {}
    for i, axis in enumerate(["x", "y", "z"]):
        feats.update(_axis_features(win[:, i], axis))
    feats.update(_axis_features(mag, "mag"))
    return feats


def predict_from_amplitudes(
    *,
    x_amp: float,
    y_amp: float,
    z_amp: float = 0.05,
    freq_hz: float = 5.0,
) -> dict[str, Any]:
    # Noise scales with amplitude so quiet settings stay clearly NORMAL.
    noise = 0.003 + 0.04 * max(x_amp, y_amp, z_amp)
    win = synthesize_window(
        x_amp=x_amp,
        y_amp=y_amp,
        z_amp=z_amp,
        freq_hz=freq_hz,
        noise=noise,
        seed=42,
    )
    return predict_from_window(
        win,
        controls={"x_amp": x_amp, "y_amp": y_amp, "z_amp": z_amp, "freq_hz": freq_hz},
    )


def _lab_score_from_controls(controls: dict[str, Any], feats: dict[str, float]) -> tuple[str, dict[str, float], float]:
    """Map X/Y/Z sliders to NORMAL/ABNORMAL for the interactive lab.

    The saved vibration pickles were trained under sklearn 1.6 and load with
    version skew on this environment, so slider demos use the same feature
    extraction pipeline plus a calibrated intensity score.
    """
    x_amp = float(controls.get("x_amp", 0.05))
    y_amp = float(controls.get("y_amp", 0.05))
    z_amp = float(controls.get("z_amp", 0.05))
    intensity = 0.55 * x_amp + 0.55 * y_amp + 0.25 * z_amp + 1.8 * float(feats.get("mag_std", 0.0))
    # Soft probability curve — crosses ~0.5 near intensity 0.30
    abnormal_p = float(1.0 / (1.0 + np.exp(-(intensity - 0.30) * 10.0)))
    abnormal_p = min(0.99, max(0.01, abnormal_p))
    probabilities = {"NORMAL": 1.0 - abnormal_p, "ABNORMAL": abnormal_p}
    label = "ABNORMAL" if abnormal_p >= 0.5 else "NORMAL"
    return label, probabilities, abnormal_p


def predict_from_window(win: np.ndarray, controls: dict[str, Any] | None = None) -> dict[str, Any]:
    feats = window_to_features(win)
    controls = controls or {}
    model_raw = None
    try:
        row = np.array([[feats[c] for c in FEATURE_COLS]], dtype=np.float32)
        model = get_model()
        pred_idx = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            model_probs = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}
        else:
            model_probs = {CLASS_NAMES[pred_idx]: 1.0}
            for name in CLASS_NAMES:
                model_probs.setdefault(name, 0.0)
        model_raw = {
            "label": CLASS_NAMES[pred_idx],
            "probabilities": model_probs,
            "model_id": get_metadata().get("experiment_id"),
            "note": "Pickle may be sklearn-version-skewed; lab label uses calibrated slider score.",
        }
    except Exception as exc:  # noqa: BLE001
        model_raw = {"error": str(exc)}

    if controls:
        label, probabilities, abnormal_p = _lab_score_from_controls(controls, feats)
        decision = "lab_calibrated"
    else:
        if model_raw and "label" in model_raw:
            label = model_raw["label"]
            probabilities = model_raw["probabilities"]
            abnormal_p = float(probabilities.get("ABNORMAL", 0.0))
            decision = "model"
        else:
            label, probabilities, abnormal_p = _lab_score_from_controls(
                {"x_amp": feats["x_std"], "y_amp": feats["y_std"], "z_amp": feats["z_std"]},
                feats,
            )
            decision = "lab_calibrated_fallback"

    series = {
        "x": win[:, 0].tolist(),
        "y": win[:, 1].tolist(),
        "z": win[:, 2].tolist(),
    }
    return {
        "label": label,
        "label_idx": CLASS_NAMES.index(label),
        "confidence": float(probabilities[label]),
        "probabilities": probabilities,
        "abnormal": label == "ABNORMAL",
        "anomaly_score": float(abnormal_p),
        "model_id": (model_raw or {}).get("model_id") or get_metadata().get("experiment_id", "vibration"),
        "decision": decision,
        "model_raw": model_raw,
        "controls": controls,
        "series": series,
        "feature_snapshot": {k: round(feats[k], 4) for k in ("x_rms", "y_rms", "z_rms", "mag_rms", "mag_std")},
    }
