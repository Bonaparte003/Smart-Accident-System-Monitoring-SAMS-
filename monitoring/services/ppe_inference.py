"""PPE wearing detection — YOLOv8 fine-tuned on Construction-PPE (or compatible weights).

Maps detector classes onto SAMS HUD items:
  hard_hat ← helmet / no_helmet
  vest     ← vest  (no dedicated no_vest label in Construction-PPE)
  goggles  ← goggles / no_goggle
  gloves   ← gloves / no_gloves

When weights are missing, returns ``available=False`` so the UI stays honest
(no fake green checkmarks from mannequin toggles on the live path).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger("monitoring.ppe")

# Construction-PPE class ids (ultralytics/cfg/datasets/construction-ppe.yaml)
CLASS_HELMET = 0
CLASS_GLOVES = 1
CLASS_VEST = 2
CLASS_BOOTS = 3
CLASS_GOGGLES = 4
CLASS_PERSON = 6
CLASS_NO_HELMET = 7
CLASS_NO_GOGGLE = 8
CLASS_NO_GLOVES = 9
CLASS_NO_BOOTS = 10

PPE_ITEMS = ("hard_hat", "vest", "goggles", "gloves")
PPE_LABELS = {
    "hard_hat": "Hard hat",
    "vest": "Safety vest",
    "goggles": "Goggles",
    "gloves": "Gloves",
}

# positive / negative class ids per SAMS item
_ITEM_CLASSES: dict[str, dict[str, int | None]] = {
    "hard_hat": {"worn": CLASS_HELMET, "missing": CLASS_NO_HELMET},
    "vest": {"worn": CLASS_VEST, "missing": None},  # no no_vest in dataset
    "goggles": {"worn": CLASS_GOGGLES, "missing": CLASS_NO_GOGGLE},
    "gloves": {"worn": CLASS_GLOVES, "missing": CLASS_NO_GLOVES},
}

MODEL_ID = "yolov8n_construction_ppe"
DEFAULT_CONF = 0.35

_lock = threading.Lock()
_model = None
_load_error: str | None = None


def _project_root() -> Path:
    try:
        from django.conf import settings as dj_settings

        if dj_settings.configured:
            return Path(dj_settings.BASE_DIR)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[2]


def weights_path() -> Path:
    custom = os.environ.get("SAMS_PPE_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    return _project_root() / "models" / "vision" / "ppe" / "best.pt"


def ppe_enabled() -> bool:
    return os.environ.get("SAMS_VISION_PPE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def weights_available() -> bool:
    return weights_path().is_file()


def get_status() -> dict[str, Any]:
    return {
        "enabled": ppe_enabled(),
        "available": ppe_enabled() and weights_available() and _load_error is None,
        "model_id": MODEL_ID,
        "weights": str(weights_path()) if weights_available() else None,
        "detail": _load_error,
        "items": list(PPE_ITEMS),
        "dataset": "Ultralytics Construction-PPE (+ optional Mendeley PPE fine-tune)",
    }


def get_model():
    """Lazy-load PPE YOLO. Raises if disabled or weights missing."""
    global _model, _load_error
    if not ppe_enabled():
        raise RuntimeError("PPE detection disabled (SAMS_VISION_PPE=0)")
    if not weights_available():
        raise RuntimeError(
            f"PPE weights not found at {weights_path()}. "
            "Train with notebooks/05_vision_ppe_detection_training.ipynb"
        )
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from ultralytics import YOLO

                    _model = YOLO(str(weights_path()))
                    _load_error = None
                    logger.info("Loaded PPE model from %s", weights_path())
                except Exception as exc:  # noqa: BLE001
                    _load_error = str(exc)
                    raise
    return _model


def reload_model() -> None:
    global _model, _load_error
    with _lock:
        _model = None
        _load_error = None


def _empty_item() -> dict[str, Any]:
    return {"worn": None, "confidence": 0.0, "evidence": "none", "status": "—"}


def _decide_item(
    scores_worn: list[float],
    scores_missing: list[float],
    *,
    person_seen: bool,
    vest_mode: bool = False,
) -> dict[str, Any]:
    best_worn = max(scores_worn) if scores_worn else 0.0
    best_miss = max(scores_missing) if scores_missing else 0.0

    if best_worn >= DEFAULT_CONF and best_worn >= best_miss:
        return {
            "worn": True,
            "confidence": float(best_worn),
            "evidence": "worn_class",
            "status": "Worn",
        }
    if best_miss >= DEFAULT_CONF and best_miss > best_worn:
        return {
            "worn": False,
            "confidence": float(best_miss),
            "evidence": "missing_class",
            "status": "Missing",
        }
    if vest_mode and person_seen:
        # Construction-PPE has no no_vest — if person present and no vest box, treat as missing
        if best_worn < DEFAULT_CONF:
            return {
                "worn": False,
                "confidence": 0.45,
                "evidence": "absent_vest_heuristic",
                "status": "Missing",
            }
    if not person_seen:
        return _empty_item()
    return {
        "worn": None,
        "confidence": 0.0,
        "evidence": "uncertain",
        "status": "—",
    }


def predict_ppe(
    image,
    *,
    conf: float = DEFAULT_CONF,
    person_hint: bool | None = None,
) -> dict[str, Any]:
    """Run PPE detector on a PIL image / ndarray. Returns wearing summary.

    ``person_hint``: if True (e.g. from the person YOLO), treat a worker as present
    even when the PPE model does not emit a ``Person`` box.
    """
    if not ppe_enabled():
        return {
            "available": False,
            "model_id": MODEL_ID,
            "items": {k: _empty_item() for k in PPE_ITEMS},
            "detections": [],
            "ppe_complete": None,
            "missing": [],
            "summary": "PPE disabled",
            "source": "disabled",
        }
    if not weights_available():
        return {
            "available": False,
            "model_id": MODEL_ID,
            "items": {k: _empty_item() for k in PPE_ITEMS},
            "detections": [],
            "ppe_complete": None,
            "missing": [],
            "summary": "PPE model not trained yet",
            "source": "unavailable",
        }

    try:
        model = get_model()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "model_id": MODEL_ID,
            "items": {k: _empty_item() for k in PPE_ITEMS},
            "detections": [],
            "ppe_complete": None,
            "missing": [],
            "summary": str(exc),
            "source": "error",
        }

    results = model.predict(image, conf=conf, verbose=False)
    detections: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = {k: [] for k in PPE_ITEMS}
    buckets_miss: dict[str, list[float]] = {k: [] for k in PPE_ITEMS}
    person_seen = bool(person_hint)

    if results:
        r0 = results[0]
        names = r0.names or {}
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(clss[i])
                score = float(scores[i])
                label = names.get(cid, str(cid))
                box = [float(v) for v in xyxy[i]]
                detections.append(
                    {"label": label, "class_id": cid, "confidence": score, "xyxy": box}
                )
                if cid == CLASS_PERSON:
                    person_seen = True
                for item, mapping in _ITEM_CLASSES.items():
                    if mapping["worn"] is not None and cid == mapping["worn"]:
                        buckets[item].append(score)
                    if mapping["missing"] is not None and cid == mapping["missing"]:
                        buckets_miss[item].append(score)

    # If only PPE boxes (no Person class), still treat as person present
    if not person_seen and any(buckets[k] or buckets_miss[k] for k in PPE_ITEMS):
        person_seen = True

    items: dict[str, Any] = {}
    for item in PPE_ITEMS:
        items[item] = _decide_item(
            buckets[item],
            buckets_miss[item],
            person_seen=person_seen,
            vest_mode=(item == "vest"),
        )
        items[item]["label"] = PPE_LABELS[item]

    missing = [PPE_LABELS[k] for k, v in items.items() if v["worn"] is False]
    known = [v for v in items.values() if v["worn"] is not None]
    if not person_seen:
        summary = "—"
        complete = None
    elif not known:
        summary = "Uncertain"
        complete = None
    elif missing:
        summary = (
            f"Missing {missing[0].lower()}"
            if len(missing) == 1
            else f"{len(missing)} items missing"
        )
        complete = False
    else:
        summary = "Complete"
        complete = True

    return {
        "available": True,
        "model_id": MODEL_ID,
        "items": items,
        "detections": detections,
        "ppe_complete": complete,
        "missing": missing,
        "summary": summary,
        "person_seen": person_seen,
        "source": "vision_ppe",
        # Convenience bools for fusion / views (None → leave sim toggles)
        "hard_hat": items["hard_hat"]["worn"],
        "vest": items["vest"]["worn"],
        "goggles": items["goggles"]["worn"],
        "gloves": items["gloves"]["worn"],
    }


def wearings_for_fusion(ppe: dict[str, Any]) -> dict[str, bool] | None:
    """Return hard_hat/vest/… bools if the model made a decision; else None."""
    if not ppe.get("available") or ppe.get("source") not in {"vision_ppe"}:
        return None
    out: dict[str, bool] = {}
    for key in PPE_ITEMS:
        worn = ppe.get(key)
        if worn is None:
            return None  # incomplete → don't override sim toggles partially
        out[key] = bool(worn)
    return out
