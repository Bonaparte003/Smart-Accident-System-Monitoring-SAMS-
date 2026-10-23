"""Person distance from the time-of-flight sensor or the trained distance network.

The sensor reading is used for the person its beam (the image centre) falls
on, within the near range stored in the distance calibration. Every other
person gets the trained network (residual on the pinhole prior,
Z = Z0 * exp(delta)) from that person's box and pose.

The served network is the one with the lowest test MAE in its metadata. A
network fine-tuned on this site's own sensor/body pairs (``site/``) wins over
the public-dataset experiments.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model_loader import LazyModel

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
_CALIB_KEYS = (
    "fy_px",
    "person_height_m",
    "tof_near_m",
    "hidden",
)


@dataclass
class DistanceEstimate:
    distance_m: float
    z0_m: float
    vision_m: float
    source: str
    confidence: float | None
    method: str
    features: dict[str, float] | None = None
    body_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "distance_m": round(self.distance_m, 3),
            "z0_m": round(self.z0_m, 3),
            "vision_m": round(self.vision_m, 3),
            "source": self.source,
            "confidence": None if self.confidence is None else round(self.confidence, 3),
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


_calib: dict[str, Any] | None = None
_device = torch.device("cpu")


def _vision_dir() -> Path:
    return _project_root() / "models" / "vision" / "distance"


def calib_path() -> Path:
    return _vision_dir() / "calib.json"


def _residual_experiments() -> list[tuple[float, Path, dict[str, Any]]]:
    found = []
    for folder in sorted(_vision_dir().glob("*")):
        meta_path = folder / "metadata.json"
        if not (folder / "model.pt").is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "exp(delta)" not in str(meta.get("formula", "")) or "absolute" in folder.name:
            continue
        mae = (meta.get("metrics_test") or {}).get("mae")
        if mae is None:
            continue
        found.append((float(mae), folder, meta))
    return sorted(found, key=lambda item: item[0])


def served_model() -> dict[str, Any]:
    """Which distance weights live inference uses, and why."""
    custom = os.environ.get("SAMS_DISTANCE_WEIGHTS", "").strip()
    if custom:
        return {"path": Path(custom), "experiment_id": "custom", "mae_m": None, "hidden": None}
    site = _vision_dir() / "site"
    if (site / "model.pt").is_file():
        meta = json.loads((site / "metadata.json").read_text()) if (site / "metadata.json").is_file() else {}
        return {
            "path": site / "model.pt",
            "experiment_id": meta.get("experiment_id", "site"),
            "mae_m": (meta.get("metrics_test") or {}).get("mae"),
            "hidden": meta.get("hidden"),
        }
    ranked = _residual_experiments()
    if ranked:
        mae, folder, meta = ranked[0]
        return {"path": folder / "model.pt", "experiment_id": folder.name, "mae_m": mae, "hidden": meta.get("hidden")}
    return {"path": _vision_dir() / "distance_nn.pt", "experiment_id": "distance_nn", "mae_m": None, "hidden": None}


def weights_path() -> Path:
    return served_model()["path"]


def load_calib() -> dict[str, Any]:
    """Pinhole calibration written by the distance notebook. Nothing is filled in here."""
    global _calib
    if _calib is not None:
        return _calib
    path = calib_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Distance calibration missing at {path}. "
            "Write it with notebooks/04_vision_distance_estimation_training.ipynb"
        )
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Distance calibration at {path} could not be read") from exc
    missing = [key for key in _CALIB_KEYS if key not in data]
    if missing:
        raise ValueError(f"Distance calibration at {path} is missing {missing}")
    _calib = data
    return _calib


def pinhole_z0_m(
    height_px: float,
    *,
    image_h: float | None = None,
    fy_px: float | None = None,
    person_height_m: float | None = None,
) -> float:
    """Z0 = fy * H / h_px, with fy taken from the exported calibration.

    When the calibration records the image height it was measured at, fy is
    scaled to the frame in front of the camera. Both fy and pixel height grow
    together when a frame is only resized.
    """
    calib = load_calib()
    h_person = float(person_height_m if person_height_m is not None else calib["person_height_m"])
    h = max(1.0, float(height_px))
    fy = float(fy_px if fy_px is not None else calib["fy_px"])
    ref_h = calib.get("fy_image_h")
    if image_h and ref_h:
        fy *= float(image_h) / float(ref_h)
    return float((fy * h_person) / h)


def aim_hits(box: Sequence[float] | None, image_wh: tuple[int, int], aim: tuple[float, float] = (0.5, 0.5)) -> bool:
    """True when the sensor aim point (image centre) lies inside this person's box."""
    if not box or len(box) < 4:
        return False
    width, height = image_wh
    ax = float(aim[0]) * max(1, width)
    ay = float(aim[1]) * max(1, height)
    return float(box[0]) <= ax <= float(box[2]) and float(box[1]) <= ay <= float(box[3])


def measure_person(
    keypoints,
    image_wh: tuple[int, int],
    *,
    box: Sequence[float],
    tof_m: float | None = None,
    sensor_aimed: bool = False,
    det_conf: float = 1.0,
    kpt_conf: float | None = None,
) -> DistanceEstimate | None:
    """The sensor range when its beam is on this person, otherwise the distance network.

    ``kpt_conf`` is the pose model's joint threshold; without it the joints are not used.
    """
    body: tuple[float, float, dict[str, float], float] | None = None
    try:
        body = predict_vision_distance(list(box), image_wh, det_conf=det_conf, keypoints=keypoints, kpt_conf=kpt_conf)
    except Exception as exc:  # noqa: BLE001
        logger.warning("distance network unavailable: %s", exc)
    sensor_ok = (
        sensor_aimed
        and tof_m is not None
        and math.isfinite(float(tof_m))
        and float(tof_m) <= float(load_calib()["tof_near_m"])
    )
    if sensor_ok:
        return DistanceEstimate(
            distance_m=float(tof_m),
            z0_m=body[1] if body else float(tof_m),
            vision_m=body[0] if body else float(tof_m),
            source="tof",
            confidence=None,
            method="tof_on_person",
            features=body[2] if body else None,
            body_m=body[0] if body else None,
        )
    if body is None:
        return None
    vision_m, z0_m, named, conf = body
    return DistanceEstimate(
        distance_m=float(vision_m),
        z0_m=float(z0_m),
        vision_m=float(vision_m),
        source="vision",
        confidence=conf,
        method=f"residual_mlp:{served_model()['experiment_id']}",
        features=named,
        body_m=float(vision_m),
    )


def _midpoint(a: Sequence[float] | None, b: Sequence[float] | None, kpt_conf: float) -> tuple[float, float, float] | None:
    if a is None or b is None:
        return None
    if a[2] < kpt_conf or b[2] < kpt_conf:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, min(a[2], b[2]))


def extract_features(
    box_xyxy: Sequence[float],
    image_wh: tuple[int, int],
    *,
    det_conf: float = 1.0,
    keypoints: Sequence[Sequence[float]] | None = None,
    kpt_conf: float | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Build feature vector + Z0 from bbox (and optional COCO-17 keypoints).

    keypoints: list of (x, y, conf) in pixel coords, length 17 when present.
    A joint counts when its confidence reaches ``kpt_conf``, the pose model's
    joint threshold; without one, the keypoints are not used.
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

    if keypoints is not None and len(keypoints) >= 17 and kpt_conf is not None:
        kps = [(float(p[0]), float(p[1]), float(p[2])) for p in keypoints]
        confs = [p[2] for p in kps if p[2] >= kpt_conf]
        if confs:
            pose_conf_mean = float(sum(confs) / len(confs))
        nose = kps[_NOSE] if kps[_NOSE][2] >= kpt_conf else None
        l_sho, r_sho = kps[_L_SHO], kps[_R_SHO]
        l_hip, r_hip = kps[_L_HIP], kps[_R_HIP]
        l_ank, r_ank = kps[_L_ANK], kps[_R_ANK]
        shoulders = _midpoint(l_sho, r_sho, kpt_conf)
        hips = _midpoint(l_hip, r_hip, kpt_conf)
        ankles = _midpoint(l_ank, r_ank, kpt_conf)
        head_y = nose[1] if nose else (shoulders[1] if shoulders else None)
        foot_y = ankles[1] if ankles else None
        if head_y is not None and foot_y is not None and foot_y > head_y:
            pose_h = foot_y - head_y
            pose_height_rel = pose_h / max(1, h)
            pose_inv_height = 1.0 / max(pose_height_rel, 1e-3)
            height_px = max(height_px, pose_h)
            pose_available = 1.0
        if l_sho[2] >= kpt_conf and r_sho[2] >= kpt_conf:
            shoulder_width_rel = abs(r_sho[0] - l_sho[0]) / max(1, w)
            pose_available = 1.0
        if l_hip[2] >= kpt_conf and r_hip[2] >= kpt_conf:
            hip_width_rel = abs(r_hip[0] - l_hip[0]) / max(1, w)
            pose_available = 1.0
        if shoulders and hips:
            torso_rel = abs(hips[1] - shoulders[1]) / max(1, h)
            pose_available = 1.0

    z0 = pinhole_z0_m(height_px, image_h=h)
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


class DistanceNet(LazyModel):
    """The residual MLP that live inference serves (see ``served_model``)."""

    def __init__(self):
        super().__init__("distance")

    def load(self) -> ResidualDistanceMLP:
        calib = load_calib()
        served = served_model()
        hidden = tuple(served.get("hidden") or calib["hidden"])
        model = ResidualDistanceMLP(FEATURE_DIM, hidden)
        path = served["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Distance weights not found at {path}. "
                "Train notebooks/04_vision_distance_estimation_training.ipynb"
            )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model


_net = DistanceNet()


def get_model() -> ResidualDistanceMLP:
    return _net.get(raise_errors=True)


def reload_model() -> None:
    """Drop cached model/calib (after training / calib update)."""
    global _calib
    _net.reset()
    _calib = None


def predict_vision_distance(
    box_xyxy: Sequence[float],
    image_wh: tuple[int, int],
    *,
    det_conf: float = 1.0,
    keypoints: Sequence[Sequence[float]] | None = None,
    kpt_conf: float | None = None,
) -> tuple[float, float, dict[str, float], float]:
    """Return (vision_m, z0_m, feature_dict, nn_confidence)."""
    feats, z0, named = extract_features(
        box_xyxy, image_wh, det_conf=det_conf, keypoints=keypoints, kpt_conf=kpt_conf
    )
    model = get_model()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0)
        delta = float(model(x).item())
    vision = float(z0 * math.exp(delta))
    return vision, z0, named, float(named["det_conf"])

