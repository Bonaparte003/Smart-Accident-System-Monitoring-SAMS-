"""PPE wearing detection — YOLOv8 fine-tuned on Construction-PPE.

Each class keeps the boxes above its own threshold from the checkpoint
metadata. A gear box belongs to the person box that holds its centre. For each item,
the strongest worn-class or missing-class box on that person decides; with
no box for an item, it is not detected. Which class name means worn or
missing is the checkpoint metadata, not a table in this file.
When weights are missing, returns ``available=False`` so the UI stays blank.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from django.conf import settings

from .model_loader import YoloModel

logger = logging.getLogger("monitoring.ppe")


def _norm_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("-", " ").replace("_", " ").split())


def metadata_path() -> Path:
    return weights_path().parent / "metadata.json"


def spec_available() -> bool:
    return metadata_path().is_file()


def load_spec() -> dict[str, Any]:
    """Class map exported with the PPE checkpoint. No built-in label table."""
    path = metadata_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"PPE metadata not found at {path}. "
            "Export it with notebooks/05_vision_ppe_detection_training.ipynb"
        )
    spec = json.loads(path.read_text())
    if "items" not in spec or "conf" not in spec or "imgsz" not in spec:
        raise ValueError(f"PPE metadata at {path} is missing items, conf, or imgsz")
    return spec


def ppe_items() -> tuple[str, ...]:
    return tuple(load_spec()["items"])


def decision_conf() -> float:
    return float(load_spec()["conf"])


def model_id() -> str:
    try:
        return str(load_spec()["model_id"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "ppe"


def _item_label(key: str, item: dict[str, Any] | None = None) -> str:
    if item and item.get("label"):
        return str(item["label"])
    return str(load_spec()["items"][key]["label"])


def _person_names() -> set[str]:
    return {_norm_label(name) for name in load_spec().get("person", [])}


def _map_label(label: str) -> tuple[str, str] | None:
    """Return (item, 'worn'|'missing') using the names stored with the checkpoint."""
    norm = _norm_label(label)
    if norm in _person_names():
        return None
    for item, cfg in load_spec()["items"].items():
        missing = {_norm_label(name) for name in cfg.get("missing") or []}
        worn = {_norm_label(name) for name in cfg.get("worn") or []}
        if norm in missing:
            return item, "missing"
        if norm in worn:
            return item, "worn"
    return None


def is_worker_box(label: str, class_id: int = 0) -> bool:
    """True when this detection's name is the person class recorded in the metadata."""
    del class_id
    return _norm_label(label) in _person_names()

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
        "available": ppe_enabled() and weights_available() and spec_available() and _loader.error is None,
        "model_id": model_id(),
        "weights": str(weights_path()) if weights_available() else None,
        "detail": _loader.error,
        "items": list(load_spec()["items"]) if spec_available() else [],
        "dataset": "Ultralytics Construction-PPE (+ optional Mendeley PPE fine-tune)",
    }


_loader = YoloModel("PPE", weights_path)


def get_model():
    """Lazy-load PPE YOLO. Raises if disabled or weights missing."""
    if not ppe_enabled():
        raise RuntimeError("PPE detection disabled (SAMS_VISION_PPE=0)")
    if not weights_available():
        raise RuntimeError(
            f"PPE weights not found at {weights_path()}. "
            "Train with notebooks/05_vision_ppe_detection_training.ipynb"
        )
    return _loader.get(raise_errors=True)


def load_private_model():
    """A separate copy of the PPE checkpoint, so a clip's tracker state stays its own."""
    from ultralytics import YOLO

    return YOLO(str(weights_path()))


def reload_model() -> None:
    _loader.reset()


def _empty_item() -> dict[str, Any]:
    return {"worn": None, "confidence": 0.0, "evidence": "none", "status": "Not detected"}


def _decide_item(scores_worn: list[float], scores_missing: list[float]) -> dict[str, Any]:
    """The stronger of this person's worn and missing boxes for one item."""
    best_worn = max(scores_worn) if scores_worn else 0.0
    best_miss = max(scores_missing) if scores_missing else 0.0
    if best_worn > best_miss:
        return {"worn": True, "confidence": float(best_worn), "evidence": "worn_class", "status": "Worn"}
    if best_miss > best_worn:
        return {"worn": False, "confidence": float(best_miss), "evidence": "missing_class", "status": "Missing"}
    return _empty_item()


def _summarize(items: dict[str, Any], person_seen: bool) -> tuple[str, bool | None, list[str]]:
    """Complete only when every item was detected worn."""
    missing = [_item_label(key, item) for key, item in items.items() if item["worn"] is False]
    if not person_seen:
        return "—", None, []
    if missing:
        summary = (
            f"Missing {missing[0].lower()}"
            if len(missing) == 1
            else f"{len(missing)} items missing"
        )
        return summary, False, missing
    if any(item["worn"] is None for item in items.values()):
        return "Uncertain", None, []
    return "Complete", True, []


def assess_person(gear: list[dict[str, Any]]) -> dict[str, Any]:
    """Wearings from the gear boxes the PPE model placed on one person."""
    items = {}
    for key in ppe_items():
        worn = [float(det["confidence"]) for det in gear if det.get("item") == key and not det.get("missing")]
        missing = [float(det["confidence"]) for det in gear if det.get("item") == key and det.get("missing")]
        items[key] = {**_decide_item(worn, missing), "label": _item_label(key)}
    summary, complete, missing_labels = _summarize(items, True)
    return {
        "available": True,
        "model_id": model_id(),
        "items": items,
        "detections": list(gear),
        "ppe_complete": complete,
        "missing": missing_labels,
        "summary": summary,
        "person_seen": True,
        "source": "vision_ppe",
        **_item_payload(items),
    }


def _item_payload(items: dict[str, Any]) -> dict[str, Any]:
    """Top-level hard_hat/vest/… mirrors, for the keys this checkpoint tracks."""
    return {key: (items.get(key) or {}).get("worn") for key in ppe_items()}


def blank_wearing() -> dict[str, Any]:
    """No person in frame. Labels still come from the checkpoint metadata."""
    try:
        spec = load_spec()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _unavailable(str(exc), "unavailable")
    items = {key: {**_empty_item(), "label": cfg["label"]} for key, cfg in spec["items"].items()}
    payload = {
        "available": True,
        "model_id": spec.get("model_id", model_id()),
        "items": items,
        "detections": [],
        "ppe_complete": None,
        "missing": [],
        "summary": "—",
        "person_seen": False,
        "source": "vision_ppe",
    }
    payload.update(_item_payload(items))
    return payload


def _unavailable(summary: str, source: str) -> dict[str, Any]:
    items: dict[str, Any] = {}
    try:
        for key, cfg in load_spec()["items"].items():
            items[key] = {**_empty_item(), "label": cfg["label"]}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        items = {}
    payload = {
        "available": False,
        "model_id": model_id(),
        "items": items,
        "detections": [],
        "ppe_complete": None,
        "missing": [],
        "summary": summary,
        "person_seen": False,
        "source": source,
    }
    for key in items:
        payload[key] = None
    return payload


def reset_tracker() -> None:
    """Drop ByteTrack state on the loaded PPE model when a camera session ends."""
    model = _loader.cached
    predictor = getattr(model, "predictor", None) if model is not None else None
    for tracker in getattr(predictor, "trackers", None) or []:
        reset = getattr(tracker, "reset", None)
        if reset is not None:
            reset()


def detect_scene(
    image, *, conf: float | None = None, track: bool = False, model=None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """One pass of the PPE checkpoint. Returns worker boxes, gear boxes, and an error payload.

    Track ids, when requested, come from that checkpoint's tracker (ByteTrack
    by default, named in the metadata). Class names are the checkpoint's names
    matched to the exported item map. A person box is not gear.
    """
    if not ppe_enabled():
        return [], [], _unavailable("PPE disabled", "disabled")
    if not weights_available():
        return [], [], _unavailable("PPE model not trained yet", "unavailable")
    if not spec_available():
        return [], [], _unavailable("PPE metadata not exported yet", "unavailable")
    try:
        spec = load_spec()
        model = model if model is not None else get_model()
    except Exception as exc:  # noqa: BLE001
        return [], [], _unavailable(str(exc), "error")

    # Each class keeps boxes from its own tuned threshold (metadata ``class_conf``);
    # a class without one, or an explicit ``conf``, uses the single ``conf``.
    default_cutoff = float(spec["conf"] if conf is None else conf)
    class_conf = {} if conf is not None else {_norm_label(k): float(v) for k, v in (spec.get("class_conf") or {}).items()}
    cutoff = min([default_cutoff, *class_conf.values()])
    kwargs: dict[str, Any] = {"conf": cutoff, "imgsz": int(spec["imgsz"]), "verbose": False}
    if track:
        kwargs["persist"] = True
        kwargs["tracker"] = str(spec.get("tracker") or "bytetrack.yaml")
        results = model.track(image, **kwargs)
    else:
        results = model.predict(image, **kwargs)
    workers: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    if not results:
        return workers, detections, None
    r0 = results[0]
    names = r0.names or {}
    boxes = getattr(r0, "boxes", None)
    if boxes is None or not len(boxes):
        return workers, detections, None
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    id_tensor = getattr(boxes, "id", None)
    track_ids = id_tensor.cpu().numpy().astype(int) if id_tensor is not None else None
    for i in range(len(xyxy)):
        cid = int(clss[i])
        score = float(scores[i])
        label = str(names.get(cid, cid))
        if score < class_conf.get(_norm_label(label), default_cutoff):
            continue
        box = [float(v) for v in xyxy[i]]
        if is_worker_box(label, cid):
            workers.append(
                {
                    "box": box,
                    "score": score,
                    "label": "person",
                    "track_id": None if track_ids is None else int(track_ids[i]),
                }
            )
            continue
        mapped = _map_label(label)
        if mapped is None:
            continue
        item, polarity = mapped
        detections.append(
            {
                "label": label,
                "item": item,
                "class_id": cid,
                "confidence": score,
                "xyxy": box,
                "missing": polarity == "missing",
            }
        )
    return workers, detections, None
