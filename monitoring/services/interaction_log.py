"""Persist a live interaction when the decision actually changes."""

from __future__ import annotations

from typing import Any

from monitoring.models import InteractionEvent

_last_signature: tuple | None = None


def _selected_person(vision: dict[str, Any]) -> dict[str, Any] | None:
    people = vision.get("people") or []
    for person in people:
        if person.get("selected"):
            return person
    return people[0] if people else None


def record_interaction(payload: dict[str, Any]) -> InteractionEvent | None:
    """Write one row when the fusion class, sentence, person, or machine state changes.

    A frame where fusion gave no decision is not written.
    """
    global _last_signature
    fusion = payload.get("fusion") or {}
    if not fusion.get("label"):
        return None
    vision = payload.get("vision") or {}
    features = payload.get("features") or {}
    vibration = payload.get("vibration") or {}
    person = _selected_person(vision)
    person_id = "" if person is None else str(person.get("id") or "")
    level = str(fusion["label"])
    reason = str(fusion.get("reason") or "")
    machine_on = bool(features.get("machine_active"))
    unusual = bool(vibration.get("abnormal"))
    signature = (level, reason, person_id, machine_on, unusual)
    if signature == _last_signature:
        return None
    _last_signature = signature
    distance = vision.get("worker_distance_est")
    score = vibration.get("anomaly_score")
    if score is None:
        score = vibration.get("mag_std")
    return InteractionEvent.objects.create(
        person_id=person_id,
        level=level,
        reason=reason,
        distance_m=None if distance is None else float(distance),
        distance_source=str(vision.get("distance_source") or ""),
        machine_on=machine_on,
        vibration_score=None if score is None else float(score),
        vibration_unusual=unusual,
        approaching=bool(features.get("approaching_machine")),
        detail={
            "ppe": (person or {}).get("ppe") or vision.get("ppe"),
            "ppe_summary": (person or {}).get("ppe_summary") or vision.get("ppe_summary"),
            "zone": payload.get("zone"),
        },
    )


def reset_interaction_signature() -> None:
    global _last_signature
    _last_signature = None
