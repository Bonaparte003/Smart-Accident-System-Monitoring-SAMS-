"""Map simulation / sensor controls into fusion feature vectors."""

from __future__ import annotations

from typing import Any

from .fusion_inference import CAUTION_ZONE_M, DANGER_ZONE_M

PPE_ITEMS = ("hard_hat", "vest", "gloves", "boots")
PPE_LABELS = {
    "hard_hat": "Hard hat",
    "vest": "Safety vest",
    "gloves": "Gloves",
    "boots": "Safety shoes",
}


def build_features(
    *,
    worker_distance: float,
    approaching_machine: bool = False,
    machine_active: bool = True,
    abnormal_vibration: bool = False,
    hard_hat: bool = True,
    vest: bool = True,
    gloves: bool = True,
    boots: bool = True,
    worker_detected: bool | None = None,
    worker_speed: float | None = None,
    vision_confidence: float | None = None,
) -> dict[str, float]:
    """Derive fusion features from workcell simulation controls.

    Vision / vibration modality outputs are synthesized from controls so the
    Three.js demo can drive the real fusion classifier without live camera /
    accelerometer streams.

    Mannequin PPE wearings are tracked for the vision HUD. They are not part of
    the trained fusion feature vector (FEATURE_COLS), but missing PPE lowers
    vision confidence and is reported in the response.
    """
    dist = max(0.05, float(worker_distance))
    wearings = {
        "hard_hat": bool(hard_hat),
        "vest": bool(vest),
        "gloves": bool(gloves),
        "boots": bool(boots),
    }

    if worker_detected is None:
        worker_detected = dist <= 8.0
    detected = bool(worker_detected)

    if worker_speed is None:
        if approaching_machine:
            worker_speed = 0.9 if dist > DANGER_ZONE_M else 0.45
        else:
            worker_speed = 0.15
    worker_speed = float(worker_speed)

    missing_count = sum(1 for on in wearings.values() if not on)
    if vision_confidence is None:
        base = 0.92 if detected and dist <= CAUTION_ZONE_M else (0.78 if detected else 0.12)
        if detected and missing_count:
            base = max(0.45, base - 0.08 * missing_count)
        vision_confidence = base
    vision_confidence = float(min(0.99, max(0.05, vision_confidence)))

    inside_danger = int(detected and dist <= DANGER_ZONE_M)
    inside_caution = int(detected and dist <= CAUTION_ZONE_M)

    vibration_state = 1.0 if abnormal_vibration else 0.0
    vibration_anomaly_score = 0.82 if abnormal_vibration else 0.12

    return {
        "worker_detected": 1.0 if detected else 0.0,
        "worker_distance": dist if detected else 99.0,
        "inside_caution_zone": float(inside_caution),
        "inside_danger_zone": float(inside_danger),
        "worker_speed": worker_speed,
        "approaching_machine": 1.0 if approaching_machine else 0.0,
        "vision_confidence": vision_confidence,
        "machine_active": 1.0 if machine_active else 0.0,
        "vibration_state": vibration_state,
        "vibration_anomaly_score": vibration_anomaly_score,
        "hard_hat": 1.0 if wearings["hard_hat"] else 0.0,
        "vest": 1.0 if wearings["vest"] else 0.0,
        "gloves": 1.0 if wearings["gloves"] else 0.0,
        "boots": 1.0 if wearings["boots"] else 0.0,
        "ppe_complete": 1.0 if missing_count == 0 else 0.0,
        "ppe_missing_count": float(missing_count),
    }


def zone_label(features: dict[str, float]) -> str:
    if features["inside_danger_zone"] >= 0.5:
        return "Danger"
    if features["inside_caution_zone"] >= 0.5:
        return "Caution"
    return "Clear"


def _ppe_status(features: dict[str, float]) -> dict[str, Any]:
    detected = features["worker_detected"] >= 0.5
    items = {}
    for key in PPE_ITEMS:
        worn = features.get(key, 1.0) >= 0.5
        items[key] = {
            "label": PPE_LABELS[key],
            "worn": worn if detected else None,
            "status": ("Worn" if worn else "Missing") if detected else "—",
        }
    missing = [PPE_LABELS[k] for k, v in items.items() if v["worn"] is False]
    if not detected:
        summary = "—"
    elif not missing:
        summary = "Complete"
    elif len(missing) == 1:
        summary = f"Missing {missing[0].lower()}"
    else:
        summary = f"{len(missing)} items missing"
    return {
        "summary": summary,
        "complete": detected and not missing,
        "missing": missing,
        "items": items,
    }


def vision_hud(features: dict[str, float], ppe: dict[str, Any]) -> dict[str, Any]:
    detected = features["worker_detected"] >= 0.5
    if not detected:
        status = "No worker"
    elif ppe["complete"]:
        status = "Worker detected"
    else:
        status = ppe["summary"]
    return {
        "status": status,
        "detected": detected,
        "hard_hat": (features.get("hard_hat", 1.0) >= 0.5) if detected else None,
        "confidence": round(features["vision_confidence"], 3),
        "distance_m": round(features["worker_distance"], 2) if detected else None,
    }


def vibration_hud(features: dict[str, float]) -> dict[str, Any]:
    abnormal = features["vibration_state"] >= 0.5
    return {
        "status": "Abnormal" if abnormal else "Normal",
        "abnormal": abnormal,
        "anomaly_score": round(features["vibration_anomaly_score"], 3),
    }


def build_simulation_response(features: dict[str, float], fusion: dict[str, Any]) -> dict[str, Any]:
    ppe = _ppe_status(features)
    return {
        "features": features,
        "zones_m": {"caution": CAUTION_ZONE_M, "danger": DANGER_ZONE_M},
        "hud": {
            "vision": vision_hud(features, ppe),
            "ppe": ppe,
            "hard_hat": ppe["items"]["hard_hat"]["status"],
            "vibration": vibration_hud(features),
            "zone": zone_label(features),
            "fusion": {
                "status": fusion["label"],
                "confidence": round(fusion["confidence"], 3),
                "probabilities": {
                    k: round(v, 4) for k, v in fusion["probabilities"].items()
                },
            },
        },
        "fusion": fusion,
        "scene": {
            "worker_distance": features["worker_distance"]
            if features["worker_detected"] >= 0.5
            else None,
            "machine_active": features["machine_active"] >= 0.5,
            "approaching_machine": features["approaching_machine"] >= 0.5,
            "abnormal_vibration": features["vibration_state"] >= 0.5,
            "wearings": {k: features.get(k, 1.0) >= 0.5 for k in PPE_ITEMS},
            "ppe_complete": features.get("ppe_complete", 1.0) >= 0.5,
            "inside_caution_zone": features["inside_caution_zone"] >= 0.5,
            "inside_danger_zone": features["inside_danger_zone"] >= 0.5,
        },
    }
