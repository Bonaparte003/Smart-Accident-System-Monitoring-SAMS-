"""Pose-geometry inspired residual NN for far-range person distance.

Close range is intended for a time-of-flight sensor; vision NN covers longer range.
Z ≈ Z0 * exp(fθ(features)), where Z0 is the pinhole prior from person height in the frame.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("monitoring.distance")


def _project_root() -> Path:
    """Repo root (…/SAMS). Works with or without Django settings."""
    try:
        from django.conf import settings

        if settings.configured:
            return Path(settings.BASE_DIR)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[2]

# COCO-17 indices used when pose keypoints are available
_NOSE, _L_SHO, _R_SHO = 0, 5, 6
_L_HIP, _R_HIP = 11, 12
_L_ANK, _R_ANK = 15, 16

FEATURE_NAMES: list[str] = [
    "rel_h",
    "rel_w",
    "aspect",
    "cx_norm",
    "cy_norm",
    "bottom_y_norm",
    "inv_rel_h",
    "log_z0",
    "det_conf",
    "pose_available",
    "pose_height_rel",
    "shoulder_width_rel",
    "hip_width_rel",
    "torso_rel",
    "pose_conf_mean",
    "pose_inv_height",
]

FEATURE_DIM = len(FEATURE_NAMES)
DEFAULT_CALIB = {
    "fy_px": 720.0,
    "person_height_m": 1.70,
    "min_dist_m": 0.5,
    "max_dist_m": 12.0,
    "tof_near_m": 1.5,
    "tof_far_m": 2.5,
    "hidden": [64, 32],
}


@dataclass
class DistanceEstimate:
    distance_m: float
    z0_m: float
    vision_m: float
    source: str  # vision_nn | tof | blend | heuristic
    confidence: float
    method: str
    features: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "distance_m": round(self.distance_m, 3),
            "z0_m": round(self.z0_m, 3),
            "vision_m": round(self.vision_m, 3),
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "method": self.method,
        }
        if self.features is not None:
            out["features"] = {k: round(float(v), 4) for k, v in self.features.items()}
        return out


class ResidualDistanceMLP(nn.Module):
    """Predicts log residual δ so Z = Z0 * exp(δ)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: Sequence[int] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Near-identity bootstrap: residual ≈ 0 → Z ≈ Z0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


_lock = threading.Lock()
_model: ResidualDistanceMLP | None = None
_calib: dict[str, Any] | None = None
_device = torch.device("cpu")


def _vision_dir() -> Path:
    return _project_root() / "models" / "vision" / "distance"


def calib_path() -> Path:
    return _vision_dir() / "calib.json"


def weights_path() -> Path:
    return _vision_dir() / "distance_nn.pt"


def load_calib() -> dict[str, Any]:
    global _calib
    if _calib is not None:
        return _calib
    path = calib_path()
    data = dict(DEFAULT_CALIB)
    if path.is_file():
        try:
            with path.open() as f:
                data.update(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("distance calib load failed: %s", exc)
    _calib = data
    return _calib


def save_default_calib(path: Path | None = None) -> Path:
    out = path or calib_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(DEFAULT_CALIB)
    payload["feature_names"] = FEATURE_NAMES
    payload["feature_dim"] = FEATURE_DIM
    payload["formula"] = "Z = Z0 * exp(f_theta(x)), Z0 = fy * H / h_px"
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    return out


def pinhole_z0_m(
    height_px: float,
    *,
    fy_px: float | None = None,
    person_height_m: float | None = None,
    min_dist_m: float | None = None,
    max_dist_m: float | None = None,
) -> float:
    """Classical prior: Z0 = (fy * H_person) / h_pixels."""
    calib = load_calib()
    fy = float(fy_px if fy_px is not None else calib["fy_px"])
    h_person = float(person_height_m if person_height_m is not None else calib["person_height_m"])
    z_min = float(min_dist_m if min_dist_m is not None else calib["min_dist_m"])
    z_max = float(max_dist_m if max_dist_m is not None else calib["max_dist_m"])
    h = max(1.0, float(height_px))
    z0 = (fy * h_person) / h
    return float(min(z_max, max(z_min, z0)))


def _midpoint(a: Sequence[float] | None, b: Sequence[float] | None) -> tuple[float, float, float] | None:
    if a is None or b is None:
        return None
    if a[2] < 0.3 or b[2] < 0.3:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, min(a[2], b[2]))


def extract_features(
    box_xyxy: Sequence[float],
    image_wh: tuple[int, int],
    *,
    det_conf: float = 1.0,
    keypoints: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Build feature vector + Z0 from bbox (and optional COCO-17 keypoints).

    keypoints: list of (x, y, conf) in pixel coords, length 17 when present.
    """
    w, h = image_wh
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    rel_h = bh / max(1, h)
    rel_w = bw / max(1, w)
    aspect = bh / bw
    cx_norm = ((x1 + x2) * 0.5) / max(1, w)
    cy_norm = ((y1 + y2) * 0.5) / max(1, h)
    bottom_y_norm = y2 / max(1, h)

    height_px = bh
    pose_available = 0.0
    pose_height_rel = 0.0
    shoulder_width_rel = 0.0
    hip_width_rel = 0.0
    torso_rel = 0.0
    pose_conf_mean = 0.0
    pose_inv_height = 0.0

    if keypoints is not None and len(keypoints) >= 17:
        kps = [(float(p[0]), float(p[1]), float(p[2])) for p in keypoints]
        confs = [p[2] for p in kps if p[2] >= 0.3]
        if confs:
            pose_conf_mean = float(sum(confs) / len(confs))
        nose = kps[_NOSE] if kps[_NOSE][2] >= 0.3 else None
        l_sho, r_sho = kps[_L_SHO], kps[_R_SHO]
        l_hip, r_hip = kps[_L_HIP], kps[_R_HIP]
        l_ank, r_ank = kps[_L_ANK], kps[_R_ANK]
        shoulders = _midpoint(l_sho, r_sho)
        hips = _midpoint(l_hip, r_hip)
        ankles = _midpoint(l_ank, r_ank)
        head_y = nose[1] if nose else (shoulders[1] if shoulders else None)
        foot_y = ankles[1] if ankles else None
        if head_y is not None and foot_y is not None and foot_y > head_y:
            pose_h = foot_y - head_y
            pose_height_rel = pose_h / max(1, h)
            pose_inv_height = 1.0 / max(pose_height_rel, 1e-3)
            height_px = max(height_px, pose_h)
            pose_available = 1.0
        if l_sho[2] >= 0.3 and r_sho[2] >= 0.3:
            shoulder_width_rel = abs(r_sho[0] - l_sho[0]) / max(1, w)
            pose_available = 1.0
        if l_hip[2] >= 0.3 and r_hip[2] >= 0.3:
            hip_width_rel = abs(r_hip[0] - l_hip[0]) / max(1, w)
            pose_available = 1.0
        if shoulders and hips:
            torso_rel = abs(hips[1] - shoulders[1]) / max(1, h)
            pose_available = 1.0

    z0 = pinhole_z0_m(height_px)
    inv_rel_h = 1.0 / max(rel_h, 1e-3)
    feats = np.array(
        [
            rel_h,
            rel_w,
            aspect,
            cx_norm,
            cy_norm,
            bottom_y_norm,
            inv_rel_h,
            math.log(max(z0, 1e-3)),
            float(min(1.0, max(0.0, det_conf))),
            pose_available,
            pose_height_rel,
            shoulder_width_rel,
            hip_width_rel,
            torso_rel,
            pose_conf_mean,
            pose_inv_height,
        ],
        dtype=np.float32,
    )
    named = {FEATURE_NAMES[i]: float(feats[i]) for i in range(FEATURE_DIM)}
    return feats, z0, named


def get_model() -> ResidualDistanceMLP:
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        calib = load_calib()
        hidden = tuple(calib.get("hidden", DEFAULT_CALIB["hidden"]))
        model = ResidualDistanceMLP(FEATURE_DIM, hidden)
        path = weights_path()
        if path.is_file():
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                logger.info("Loaded distance NN from %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("distance NN load failed (%s); using identity residual", exc)
        else:
            logger.info("No distance_nn.pt yet — residual ≈ 0 (Z ≈ Z0 pinhole)")
        model.eval()
        _model = model
        return _model


def reload_model() -> None:
    """Drop cached model/calib (after training / calib update)."""
    global _model, _calib
    with _lock:
        _model = None
        _calib = None


def predict_vision_distance(
    box_xyxy: Sequence[float],
    image_wh: tuple[int, int],
    *,
    det_conf: float = 1.0,
    keypoints: Sequence[Sequence[float]] | None = None,
) -> tuple[float, float, dict[str, float], float]:
    """Return (vision_m, z0_m, feature_dict, nn_confidence)."""
    feats, z0, named = extract_features(
        box_xyxy, image_wh, det_conf=det_conf, keypoints=keypoints
    )
    calib = load_calib()
    z_min = float(calib["min_dist_m"])
    z_max = float(calib["max_dist_m"])
    model = get_model()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0)
        delta = float(model(x).item())
    # Soft clamp residual so a wild net cannot explode
    delta = max(-1.5, min(1.5, delta))
    vision = z0 * math.exp(delta)
    vision = float(min(z_max, max(z_min, vision)))
    # Confidence: higher when pose present and Z0 in mid/far band
    conf = 0.55 + 0.15 * named["pose_available"] + 0.15 * min(1.0, named["det_conf"])
    if vision >= float(calib["tof_near_m"]):
        conf += 0.1
    conf = float(min(0.95, conf))
    return vision, z0, named, conf


def fuse_with_tof(
    vision_m: float,
    *,
    z0_m: float,
    vision_conf: float,
    features: dict[str, float] | None = None,
    tof_m: float | None = None,
    tof_valid: bool | None = None,
) -> DistanceEstimate:
    """Gate: ToF for near, NN for far, blend in between."""
    calib = load_calib()
    d_near = float(calib["tof_near_m"])
    d_far = float(calib["tof_far_m"])
    method = "residual_mlp_pinhole"

    if tof_valid is None:
        tof_valid = tof_m is not None and math.isfinite(float(tof_m)) and float(tof_m) > 0.05
    if tof_valid and tof_m is not None:
        tof = float(tof_m)
        if tof <= d_near:
            return DistanceEstimate(
                distance_m=tof,
                z0_m=z0_m,
                vision_m=vision_m,
                source="tof",
                confidence=0.95,
                method=method,
                features=features,
            )
        if tof <= d_far:
            # Smooth blend: weight ToF more when closer
            t = (tof - d_near) / max(1e-3, d_far - d_near)
            w_tof = 1.0 - t
            blended = w_tof * tof + t * vision_m
            return DistanceEstimate(
                distance_m=float(blended),
                z0_m=z0_m,
                vision_m=vision_m,
                source="blend",
                confidence=float(0.7 * w_tof + vision_conf * t),
                method=method,
                features=features,
            )

    return DistanceEstimate(
        distance_m=vision_m,
        z0_m=z0_m,
        vision_m=vision_m,
        source="vision_nn",
        confidence=vision_conf,
        method=method,
        features=features,
    )


def estimate_person_distance(
    box_xyxy: Sequence[float],
    image_wh: tuple[int, int],
    *,
    det_conf: float = 1.0,
    keypoints: Sequence[Sequence[float]] | None = None,
    tof_m: float | None = None,
) -> DistanceEstimate:
    vision, z0, feats, conf = predict_vision_distance(
        box_xyxy, image_wh, det_conf=det_conf, keypoints=keypoints
    )
    return fuse_with_tof(
        vision, z0_m=z0, vision_conf=conf, features=feats, tof_m=tof_m
    )


def heuristic_distance_m(box_xyxy: Sequence[float], image_h: int) -> float:
    """Legacy fallback (bbox relative height)."""
    calib = load_calib()
    y1, y2 = float(box_xyxy[1]), float(box_xyxy[3])
    rel_h = max(1e-3, (y2 - y1) / max(1, image_h))
    dist = 0.65 / rel_h
    return float(min(calib["max_dist_m"], max(calib["min_dist_m"], dist)))
