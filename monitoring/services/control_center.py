"""The SAMS decision: the fusion model's label for what the other models measured.

Every fusion input comes from a model output or a sensor reading in this
frame. When one of the inputs the fusion model was trained on was not
measured, there is no decision, and the reason names what is missing.
"""

from __future__ import annotations

from typing import Any

from hardware.stream import apply_risk_actuators, get_stream

from .fusion_inference import absent_worker_values, get_feature_cols, predict_fusion, zones_m

_INPUT_NAMES = {
    "ppe_missing": "PPE",
    "worker_detected": "person",
    "vision_confidence": "person",
    "worker_distance": "distance",
    "inside_caution_zone": "distance",
    "inside_danger_zone": "distance",
    "worker_speed": "worker motion",
    "approaching_machine": "worker motion",
    "approach_velocity": "worker motion",
    "machine_active": "machine state",
    "vibration_state": "vibration",
    "vibration_anomaly_score": "vibration",
    "fire_score": "fire/smoke",
    "smoke_score": "fire/smoke",
    "fall_detected": "fall",
}

_last_label: str | None = None


def measured_inputs(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine_on: bool | None,
    absent_worker: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fusion inputs that a model or sensor produced for this frame. Nothing else.

    With nobody in the frame, the person inputs take ``absent_worker``: the
    values the fusion model was trained with for an empty frame.
    """
    vision = vision or {}
    out: dict[str, float] = {"worker_detected": 1.0 if vision.get("worker_detected") else 0.0}
    if not vision.get("worker_detected"):
        out.update(absent_worker or {})
    else:
        if vision.get("confidence") is not None:
            out["vision_confidence"] = float(vision["confidence"])
        distance = vision.get("worker_distance_est")
        if distance is not None:
            zones = zones_m()
            out["worker_distance"] = float(distance)
            out["inside_caution_zone"] = 1.0 if float(distance) <= zones["caution"] else 0.0
            out["inside_danger_zone"] = 1.0 if float(distance) <= zones["danger"] else 0.0
        if vision.get("worker_speed") is not None:
            out["worker_speed"] = float(vision["worker_speed"])
        if vision.get("approaching") is not None:
            out["approaching_machine"] = 1.0 if vision["approaching"] else 0.0
        if vision.get("approach_velocity") is not None:
            out["approach_velocity"] = float(vision["approach_velocity"])
        ppe = vision.get("ppe") if isinstance(vision.get("ppe"), dict) else {}
        if ppe.get("available"):
            out["ppe_missing"] = 1.0 if vision.get("ppe_complete") is False else 0.0
    if machine_on is not None:
        out["machine_active"] = 1.0 if machine_on else 0.0
    if vibration:
        out["vibration_state"] = 1.0 if vibration.get("abnormal") else 0.0
        out["vibration_anomaly_score"] = float(vibration["anomaly_score"])
    hazards = vision.get("hazards") if isinstance(vision.get("hazards"), dict) else {}
    if hazards.get("available"):
        out["fire_score"] = float(hazards.get("fire_score") or 0.0)
        out["smoke_score"] = float(hazards.get("smoke_score") or 0.0)
    if vision.get("falls_available"):
        out["fall_detected"] = 1.0 if vision.get("fall_detected") else 0.0
    return out


def decide(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine_on: bool | None,
) -> dict[str, Any]:
    """The fusion model's label, or no label when one of its inputs was not measured."""
    features = measured_inputs(vision, vibration, machine_on=machine_on, absent_worker=absent_worker_values())
    cols = get_feature_cols()
    missing = sorted({_INPUT_NAMES.get(col, col) for col in cols if col not in features})
    if missing:
        return {
            "label": None,
            "label_idx": None,
            "confidence": None,
            "probabilities": None,
            "model_id": None,
            "reason": "No decision · not measured: " + ", ".join(missing),
            "missing": missing,
            "features": features,
        }
    fusion = predict_fusion({col: features[col] for col in cols})
    return {**fusion, "reason": "", "missing": [], "features": features}


def zone_label(features: dict[str, float]) -> str:
    if "worker_distance" not in features:
        return "—"
    if features.get("inside_danger_zone", 0.0) >= 0.5:
        return "Danger"
    if features.get("inside_caution_zone", 0.0) >= 0.5:
        return "Caution"
    return "Clear"


def assemble(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine: dict[str, Any] | None = None,
    actuate: bool = False,
) -> dict[str, Any]:
    """Fusion decision for one vision frame, one vibration window, and the machine-state label."""
    global _last_label
    vision = vision or {}
    stream = get_stream()
    decision = decide(vision, vibration, machine_on=machine["running"] if machine else None)
    features = decision.pop("features")
    label = decision["label"]
    if actuate and label != _last_label:
        _last_label = label
        apply_risk_actuators(label, buzzer=stream.buzzer_auto, leds=stream.leds_auto)

    detected = bool(vision.get("worker_detected"))
    ppe = vision.get("ppe") if isinstance(vision.get("ppe"), dict) else None
    distance_m = features.get("worker_distance")
    return {
        "fusion": decision,
        "features": features,
        "zone": zone_label(features),
        "vision": {
            "label": vision.get("label") or ("Person" if detected else "No person"),
            "confidence": vision.get("confidence"),
            "worker_detected": detected,
            "person_count": vision.get("person_count"),
            "people": vision.get("people") or [],
            "worker_distance_est": distance_m,
            "distance_source": vision.get("distance_source") if distance_m is not None else None,
            "vision_distance_m": vision.get("vision_distance_m"),
            "ppe": ppe,
            "ppe_summary": vision.get("ppe_summary") or (ppe or {}).get("summary"),
            "fire_score": vision.get("fire_score"),
            "smoke_score": vision.get("smoke_score"),
            "fall_detected": vision.get("fall_detected"),
            "thumbnail": vision.get("thumbnail") or vision.get("annotated_thumbnail"),
        },
        "vibration": None
        if vibration is None
        else {
            "label": vibration.get("label"),
            "confidence": vibration.get("confidence"),
            "abnormal": bool(vibration.get("abnormal")),
            "anomaly_score": vibration.get("anomaly_score"),
            "ready": True,
            "declared": bool(vibration.get("declared")),
        },
        "machine": None
        if machine is None
        else {
            "label": machine.get("label"),
            "confidence": machine.get("confidence"),
            "model_id": machine.get("model_id"),
            "declared": bool(machine.get("declared")),
        },
    }
