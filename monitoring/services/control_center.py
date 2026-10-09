"""Join live vision, distance, and vibration into the SAMS fusion decision."""

from __future__ import annotations

import threading
import time
from typing import Any

from hardware.stream import apply_risk_actuators, get_stream

from .feature_builder import build_features, zone_label
from .fusion_inference import predict_fusion

_MEASURED_SOURCES = {"tof", "vision", "pinhole"}

_lock = threading.Lock()
_track = {"dist": None, "t": None}
_last_label: str | None = None


def observe_motion(distance_m: float | None, detected: bool) -> tuple[float, bool]:
    """Speed (m/s) and whether the worker moved closer since the last sample."""
    now = time.monotonic()
    speed = 0.0
    approaching = False
    with _lock:
        prev_d = _track["dist"]
        prev_t = _track["t"]
        if detected and distance_m is not None and prev_d is not None and prev_t is not None:
            dt = max(now - float(prev_t), 0.05)
            delta = float(prev_d) - float(distance_m)
            speed = min(3.0, abs(delta) / dt)
            approaching = delta > 0.05
        if detected and distance_m is not None:
            _track["dist"] = float(distance_m)
            _track["t"] = now
    return speed, approaching


def measured_distance(vision: dict[str, Any] | None) -> float | None:
    """Metres from the sensor or the distance network. Anything else is absent."""
    vision = vision or {}
    source = vision.get("distance_source")
    measured = vision.get("worker_distance_est")
    if measured is None:
        measured = vision.get("distance_m")
    if source not in _MEASURED_SOURCES or measured is None:
        return None
    try:
        value = float(measured)
    except (TypeError, ValueError):
        return None
    return value


def classify(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine_on: bool,
    speed: float = 0.0,
    approaching: bool = False,
) -> dict[str, Any]:
    """Fusion-model label.

    A person with no measured distance is DEGRADED. The model is not given a
    stand-in distance for that person. With no person in frame, distance 99
    is the training value for "nobody", and the model still scores vibration.
    """
    vision = vision or {}
    detected = bool(vision.get("worker_detected"))
    distance_m = measured_distance(vision) if detected else None
    if detected and distance_m is None:
        return {
            "label": "DEGRADED",
            "label_idx": None,
            "confidence": None,
            "probabilities": {"NORMAL": 0.0, "CAUTION": 0.0, "DANGER": 0.0},
            "model_id": None,
            "model_label": None,
            "reason": "Person seen, distance unavailable.",
            "distance_m": None,
        }

    ppe = vision.get("ppe") if isinstance(vision.get("ppe"), dict) else None
    confidence = vision.get("confidence")
    abnormal = bool(vibration and vibration.get("abnormal"))
    vib_score = float(vibration["anomaly_score"]) if vibration and vibration.get("anomaly_score") is not None else None
    features = build_features(
        worker_distance=distance_m if distance_m is not None else 99.0,
        approaching_machine=approaching,
        machine_active=machine_on,
        abnormal_vibration=abnormal,
        hard_hat=_ppe_bool(ppe, "hard_hat"),
        vest=_ppe_bool(ppe, "vest"),
        gloves=_ppe_bool(ppe, "gloves"),
        boots=_ppe_bool(ppe, "boots"),
        worker_detected=detected,
        worker_speed=speed,
        vision_confidence=float(confidence) if confidence is not None else 0.0,
    )
    if vib_score is not None:
        features["vibration_anomaly_score"] = vib_score
        features["vibration_state"] = 1.0 if abnormal else 0.0
    fusion = predict_fusion(features)
    return {
        **fusion,
        "model_label": fusion["label"],
        "reason": "",
        "distance_m": distance_m,
        "features": features,
    }


LEVEL_RANK = {"NORMAL": 0, "DEGRADED": 1, "CAUTION": 2, "DANGER": 3}


def scene_events(
    *,
    fallen_ids: list[Any] | None = None,
    fire: bool = False,
    smoke: bool = False,
    behaviour: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Scene findings the fusion model has no input for, each with the level it demands."""
    events: list[tuple[str, str]] = []
    fallen_ids = list(fallen_ids or [])
    if fallen_ids:
        named = [f"person {pid}" for pid in fallen_ids if pid is not None]
        who = ", ".join(named) if named else "a person"
        events.append(("DANGER", f"Fall: {who} down"))
    if fire:
        events.append(("DANGER", "Fire detected"))
    if smoke:
        events.append(("CAUTION", "Smoke detected"))
    if behaviour and behaviour.get("unsafe"):
        events.append(("CAUTION", f"Unsafe behaviour: {behaviour['label']}"))
    return events


def apply_scene_events(level: str, reason: str, events: list[tuple[str, str]]) -> tuple[str, str]:
    """Raise the fusion level to the worst scene finding. Findings lead the reason."""
    if not events:
        return level, reason
    worst = max(events, key=lambda event: LEVEL_RANK.get(event[0], 0))[0]
    if LEVEL_RANK.get(worst, 0) > LEVEL_RANK.get(level, 0):
        level = worst
    found = " · ".join(text for _level, text in events)
    return level, f"{found} · {reason}" if reason else found


def decide_alert(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine_on: bool | None = None,
) -> str:
    """Class from the fusion model, for the buzzer and LEDs."""
    return _live_decision(vision, vibration, machine_on=machine_on)["label"]


def _ppe_bool(ppe: dict[str, Any] | None, key: str) -> bool:
    """True only when the detector saw the item being worn.

    Unknown stays False here so fusion features do not record gear that was
    never seen. The alert itself still ignores unknown and buzzes only on an
    explicit miss (see decide_alert). A disabled detector assumes worn.
    """
    if not ppe or not ppe.get("available"):
        return True
    return ppe.get(key) is True


def _live_decision(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    machine_on: bool | None = None,
) -> dict[str, Any]:
    vision = vision or {}
    detected = bool(vision.get("worker_detected"))
    distance_m = measured_distance(vision) if detected else None
    speed, approaching = observe_motion(distance_m, detected)
    if machine_on is None:
        machine_on = get_stream().machine_on
    return classify(
        vision,
        vibration,
        machine_on=machine_on,
        speed=speed,
        approaching=approaching,
    )


def assemble(
    vision: dict[str, Any] | None,
    vibration: dict[str, Any] | None,
    *,
    actuate: bool = False,
) -> dict[str, Any]:
    """Fusion-model decision for one vision frame and one vibration window."""
    vision = vision or {}
    detected = bool(vision.get("worker_detected"))
    source = vision.get("distance_source")
    confidence = vision.get("confidence")
    ppe = vision.get("ppe") if isinstance(vision.get("ppe"), dict) else None
    abnormal = bool(vibration and vibration.get("abnormal"))
    decision = _live_decision(vision, vibration)
    distance_m = decision.get("distance_m")
    features = decision.get("features") or {
        "worker_detected": 1.0 if detected else 0.0,
        "inside_caution_zone": 0.0,
        "inside_danger_zone": 0.0,
        "worker_speed": 0.0,
        "approaching_machine": 0.0,
        "vision_confidence": float(confidence) if confidence is not None else 0.0,
        "machine_active": 1.0 if get_stream().machine_on else 0.0,
        "vibration_state": 1.0 if abnormal else 0.0,
        "vibration_anomaly_score": float(vibration["anomaly_score"])
        if vibration and vibration.get("anomaly_score") is not None
        else 0.0,
    }
    fusion = {key: value for key, value in decision.items() if key not in {"features", "distance_m"}}
    label = str(fusion["label"])
    if actuate:
        global _last_label
        if label != _last_label:
            _last_label = label
            stream = get_stream()
            apply_risk_actuators(
                label,
                buzzer=stream.buzzer_auto,
                leds=stream.leds_auto,
            )

    return {
        "fusion": fusion,
        "features": {
            **{k: features[k] for k in (
                "worker_detected",
                "inside_caution_zone",
                "inside_danger_zone",
                "worker_speed",
                "approaching_machine",
                "vision_confidence",
                "machine_active",
                "vibration_state",
                "vibration_anomaly_score",
            )},
            "worker_distance": distance_m,
        },
        "zone": "—" if detected and distance_m is None else zone_label(features),
        "vision": {
            "label": vision.get("label") or ("Person" if detected else "No person"),
            "confidence": confidence,
            "worker_detected": detected,
            "person_count": vision.get("person_count"),
            "people": vision.get("people") or [],
            "worker_distance_est": distance_m,
            "distance_source": source if distance_m is not None else None,
            "vision_distance_m": vision.get("vision_distance_m"),
            "ppe": ppe,
            "ppe_summary": vision.get("ppe_summary") or (ppe or {}).get("summary"),
            "thumbnail": vision.get("thumbnail") or vision.get("annotated_thumbnail"),
        },
        "vibration": None
        if vibration is None
        else {
            "label": vibration.get("label"),
            "confidence": vibration.get("confidence"),
            "abnormal": abnormal,
            "anomaly_score": vibration.get("anomaly_score"),
            "ready": True,
        },
    }
