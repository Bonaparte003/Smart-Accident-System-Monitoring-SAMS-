"""PPE wearing detection — YOLOv8 fine-tuned on Construction-PPE.

Gear is accepted only inside that person's body region:
  head  - hard hat
  torso - vest
  hands - gloves (both wrists)
  feet  - shoes (both ankles)

Which class name means worn or missing is the checkpoint metadata, not a
table in this file. A class the model does not have (there is no no-vest
class) is never reported as missing.

A hidden wrist or ankle stays "Not visible". It is not reported as missing.
Across a person's track, a part that appears later can confirm the gear.
A part that is still out of frame never becomes a missing flag.
When weights are missing, returns ``available=False`` so the UI stays blank.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger("monitoring.ppe")

# Item names, display labels, confidence, and image size come from
# models/vision/ppe/metadata.json, written by
# notebooks/05_vision_ppe_detection_training.ipynb from the checkpoint's classes.
# COCO-17 indices used to place each region on the person.
_HEAD_IDX = (0, 1, 2, 3, 4)
_L_SHO, _R_SHO = 5, 6
_L_ELB, _R_ELB = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_ANK, _R_ANK = 15, 16
_KP_MIN = 0.3
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

_lock = threading.Lock()
_model = None
_load_error: str | None = None
_vote_lock = threading.Lock()
# track id - item - recent worn votes
_votes: dict[str, dict[str, list[bool | None]]] = {}


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
        "available": ppe_enabled() and weights_available() and spec_available() and _load_error is None,
        "model_id": model_id(),
        "weights": str(weights_path()) if weights_available() else None,
        "detail": _load_error,
        "items": list(load_spec()["items"]) if spec_available() else [],
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
) -> dict[str, Any]:
    """A wearing is whatever class the PPE model actually scored above its own confidence."""
    cutoff = decision_conf()
    best_worn = max(scores_worn) if scores_worn else 0.0
    best_miss = max(scores_missing) if scores_missing else 0.0

    if best_worn >= cutoff and best_worn >= best_miss:
        return {
            "worn": True,
            "confidence": float(best_worn),
            "evidence": "worn_class",
            "status": "Worn",
        }
    if best_miss >= cutoff and best_miss > best_worn:
        return {
            "worn": False,
            "confidence": float(best_miss),
            "evidence": "missing_class",
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


def _summarize(items: dict[str, Any], person_seen: bool) -> tuple[str, bool | None, list[str]]:
    """Complete only when every item is worn. A hidden part is not a miss."""
    missing = [_item_label(key, item) for key, item in items.items() if item["worn"] is False]
    hidden = [key for key, item in items.items() if item.get("evidence") == "not_visible"]
    undecided = [
        key
        for key, item in items.items()
        if item["worn"] is None and item.get("evidence") != "not_visible"
    ]
    if not person_seen:
        return "—", None, []
    if missing:
        summary = (
            f"Missing {missing[0].lower()}"
            if len(missing) == 1
            else f"{len(missing)} items missing"
        )
        return summary, False, missing
    if undecided:
        return "Uncertain", None, []
    if hidden:
        return "Not fully visible", None, []
    return "Complete", True, []


def worker_has_head_or_shoulders(keypoints) -> bool:
    """A worker pose shows a head or a shoulder. Hips are not required.

    A limb with neither is not a worker. An empty pose is not a decision;
    the caller keeps the construction person box when pose found nothing.
    """
    if any(_kp(keypoints, index) for index in _HEAD_IDX):
        return True
    return _kp(keypoints, _L_SHO) is not None or _kp(keypoints, _R_SHO) is not None


def _kp(keypoints, index: int) -> tuple[float, float, float] | None:
    if not keypoints or index >= len(keypoints) or len(keypoints[index]) < 3:
        return None
    x, y, conf = (float(keypoints[index][0]), float(keypoints[index][1]), float(keypoints[index][2]))
    if conf < _KP_MIN:
        return None
    return x, y, conf


def _box_around(points: list[tuple[float, float, float]], pad_x: float, pad_top: float, pad_bottom: float) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs) - pad_x, min(ys) - pad_top, max(xs) + pad_x, max(ys) + pad_bottom]


def body_regions(
    keypoints,
    person_box: list[float] | None,
) -> dict[str, Any]:
    """Rectangles for head, torso, each wrist, and each ankle.

    A side is None when that keypoint was not confident enough to place it.
    """
    regions: dict[str, Any] = {"head": None, "torso": None, "hands": [None, None], "feet": [None, None]}
    head_pts = [p for i in _HEAD_IDX if (p := _kp(keypoints, i))]
    if head_pts:
        face_w = max(12.0, max(p[0] for p in head_pts) - min(p[0] for p in head_pts))
        face_h = max(12.0, max(p[1] for p in head_pts) - min(p[1] for p in head_pts))
        regions["head"] = _box_around(head_pts, face_w * 0.8, face_h * 1.6, face_h * 0.35)

    shoulders = [p for i in (_L_SHO, _R_SHO) if (p := _kp(keypoints, i))]
    hips = [p for i in (_L_HIP, _R_HIP) if (p := _kp(keypoints, i))]
    if shoulders and hips:
        torso_pts = shoulders + hips
        span_x = max(8.0, max(p[0] for p in torso_pts) - min(p[0] for p in torso_pts))
        span_y = max(8.0, max(p[1] for p in torso_pts) - min(p[1] for p in torso_pts))
        regions["torso"] = _box_around(torso_pts, span_x * 0.25, span_y * 0.15, span_y * 0.15)

    height = 80.0
    if person_box and len(person_box) >= 4:
        height = max(32.0, float(person_box[3]) - float(person_box[1]))
    foot_pad = max(14.0, height * 0.09)
    for slot, (wrist_i, elbow_i) in enumerate(((_L_WRIST, _L_ELB), (_R_WRIST, _R_ELB))):
        point = _kp(keypoints, wrist_i)
        if point:
            regions["hands"][slot] = _hand_region(point, _kp(keypoints, elbow_i), height)
    for slot, index in enumerate((_L_ANK, _R_ANK)):
        point = _kp(keypoints, index)
        if point:
            regions["feet"][slot] = [
                point[0] - foot_pad,
                point[1] - foot_pad * 0.4,
                point[0] + foot_pad,
                point[1] + foot_pad * 1.3,
            ]
    return regions


def _center_inside(box: list[float], region: list[float] | None) -> bool:
    if not region:
        return False
    cx = (float(box[0]) + float(box[2])) * 0.5
    cy = (float(box[1]) + float(box[3])) * 0.5
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _overlaps_region(box: list[float], region: list[float] | None) -> bool:
    """A glove box counts when its center is on the hand, or a fair part of the box is."""
    if not region:
        return False
    if _center_inside(box, region):
        return True
    ix1, iy1 = max(float(box[0]), float(region[0])), max(float(box[1]), float(region[1]))
    ix2, iy2 = min(float(box[2]), float(region[2])), min(float(box[3]), float(region[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
    return inter / area >= 0.2


def _hand_region(wrist: tuple[float, float, float], elbow: tuple[float, float, float] | None, height: float) -> list[float]:
    """Cover the hand beyond the wrist. Glove boxes rarely sit on the joint itself."""
    pad = max(22.0, height * 0.12)
    if elbow is None:
        return [wrist[0] - pad, wrist[1] - pad * 0.7, wrist[0] + pad, wrist[1] + pad * 1.3]
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    tip_x = wrist[0] + dx * 0.85
    tip_y = wrist[1] + dy * 0.85
    side = max(pad, length * 0.45)
    xs = (wrist[0], tip_x)
    ys = (wrist[1], tip_y)
    return [min(xs) - side, min(ys) - side * 0.7, max(xs) + side, max(ys) + side * 0.7]


def _not_visible() -> dict[str, Any]:
    return {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible"}


def _pair_decision(sides: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Both sides must be visible and worn. One explicit miss is missing."""
    visible = [side for side in sides if side is not None]
    if not visible:
        return _not_visible()
    misses = [side for side in visible if side["worn"] is False]
    if misses:
        best = max(misses, key=lambda side: side["confidence"])
        return {
            "worn": False,
            "confidence": best["confidence"],
            "evidence": best["evidence"],
            "status": "Missing",
        }
    if any(side is None for side in sides) or any(side["worn"] is None for side in visible):
        if any(side is None for side in sides) and all(side is None or side["worn"] is not None for side in sides):
            return _not_visible()
        return {"worn": None, "confidence": 0.0, "evidence": "uncertain", "status": "—"}
    best = max(visible, key=lambda side: side["confidence"])
    return {
        "worn": True,
        "confidence": best["confidence"],
        "evidence": "worn_class",
        "status": "Worn",
    }


def _intersection(box: list[float], region: list[float]) -> float:
    ix1, iy1 = max(float(box[0]), float(region[0])), max(float(box[1]), float(region[1]))
    ix2, iy2 = min(float(box[2]), float(region[2])), min(float(box[3]), float(region[3]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _claim_pair(item: str, regions: list, gear: list[dict[str, Any]], used: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Give each glove box to the one hand it overlaps most."""
    claimed: list[list[dict[str, Any]]] = [[] for _ in regions]
    for det in gear:
        if det.get("item") != item:
            continue
        best_i = None
        best_area = 0.0
        for index, region in enumerate(regions):
            if region is None or not _overlaps_region(det["xyxy"], region):
                continue
            area = _intersection(det["xyxy"], region)
            if area > best_area:
                best_area = area
                best_i = index
        if best_i is None:
            continue
        claimed[best_i].append(det)
        used.append(det)
    sides: list[dict[str, Any] | None] = []
    for index, region in enumerate(regions):
        if region is None:
            sides.append(None)
            continue
        hits = claimed[index]
        worn = [float(hit["confidence"]) for hit in hits if not hit.get("missing")]
        missing = [float(hit["confidence"]) for hit in hits if hit.get("missing")]
        sides.append(_decide_item(worn, missing, person_seen=True))
    return sides


def _scores_in(region: list[float] | None, gear: list[dict[str, Any]], item: str) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    worn: list[float] = []
    missing: list[float] = []
    used: list[dict[str, Any]] = []
    if not region:
        return worn, missing, used
    for det in gear:
        if det.get("item") != item:
            continue
        if not _overlaps_region(det["xyxy"], region):
            continue
        used.append(det)
        if det.get("missing"):
            missing.append(float(det["confidence"]))
        else:
            worn.append(float(det["confidence"]))
    return worn, missing, used


def assess_person(
    keypoints,
    person_box: list[float] | None,
    gear: list[dict[str, Any]],
    *,
    regions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one person's four regions. Gear outside those regions is ignored."""
    regions = regions if regions is not None else body_regions(keypoints, person_box)
    used: list[dict[str, Any]] = []

    def one_region(item: str, region: list[float] | None) -> dict[str, Any]:
        if region is None:
            return _not_visible()
        worn, missing, hits = _scores_in(region, gear, item)
        used.extend(hits)
        return _decide_item(worn, missing, person_seen=True)

    hands = _claim_pair("gloves", regions["hands"], gear, used)
    feet = [one_region("boots", region) if region is not None else None for region in regions["feet"]]
    gloves = _pair_decision(hands)
    boots = _pair_decision(feet)
    gloves["sides"] = hands
    boots["sides"] = feet
    items = {
        "hard_hat": one_region("hard_hat", regions["head"]),
        "vest": one_region("vest", regions["torso"]),
        "gloves": gloves,
        "boots": boots,
    }
    for key in ppe_items():
        items[key]["label"] = _item_label(key, items[key])
    summary, complete, missing = _summarize(items, True)
    return {
        "available": True,
        "model_id": model_id(),
        "items": items,
        "detections": used,
        "ppe_complete": complete,
        "missing": missing,
        "summary": summary,
        "person_seen": True,
        "source": "vision_ppe",
        "hard_hat": items["hard_hat"]["worn"],
        "vest": items["vest"]["worn"],
        "gloves": items["gloves"]["worn"],
        "boots": items["boots"]["worn"],
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
    model = _model
    predictor = getattr(model, "predictor", None) if model is not None else None
    for tracker in getattr(predictor, "trackers", None) or []:
        reset = getattr(tracker, "reset", None)
        if reset is not None:
            reset()


def detect_scene(
    image, *, conf: float | None = None, track: bool = False
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
        model = get_model()
    except Exception as exc:  # noqa: BLE001
        return [], [], _unavailable(str(exc), "error")

    cutoff = float(spec["conf"] if conf is None else conf)
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


def detect_gear(image, *, conf: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """One full-frame pass. Returns gear boxes, or an unavailable payload."""
    _workers, gear, blocked = detect_scene(image, conf=conf)
    return gear, blocked


def predict_ppe(
    image,
    *,
    conf: float | None = None,
    person_hint: bool | None = None,
    person_box: list[float] | None = None,
    keypoints=None,
) -> dict[str, Any]:
    """Detect gear on the full frame and score it only with a pose.

    Without keypoints the items stay unknown. A gear box is not a wearing
    until it sits on that person's head, torso, hand, or foot.
    """
    gear, blocked = detect_gear(image, conf=conf)
    if blocked is not None:
        return blocked
    if keypoints is not None:
        return assess_person(keypoints, person_box, gear)
    blank = blank_wearing()
    summary, complete, missing = _summarize(blank["items"], bool(person_hint))
    blank["summary"] = summary
    blank["ppe_complete"] = complete
    blank["missing"] = missing
    blank["person_seen"] = bool(person_hint)
    return blank


def reset_live_votes() -> None:
    """Drop the live-frame vote history (camera session start or stop)."""
    with _vote_lock:
        _votes.clear()


def prune_live_votes(active_ids: set[str]) -> None:
    """Forget workers who have left the frame."""
    with _vote_lock:
        for track_id in [key for key in _votes if key not in active_ids]:
            del _votes[track_id]


def _remember_value(previous: bool | None, item: dict[str, Any] | None) -> bool | None:
    """Keep a confirmed wearing. A hidden or uncertain part does not erase it."""
    if item is None or item.get("evidence") in {"not_visible", "uncertain"} or item.get("worn") is None:
        return previous
    if item.get("worn") is True:
        return True
    if item.get("worn") is False:
        return False
    return previous


def _remember_item(previous: Any, item: dict[str, Any]) -> Any:
    sides = item.get("sides")
    if isinstance(sides, list):
        remembered = list(previous) if isinstance(previous, list) else [None, None]
        while len(remembered) < len(sides):
            remembered.append(None)
        for index, side in enumerate(sides):
            remembered[index] = _remember_value(remembered[index], side)
        return remembered[: len(sides)]
    return _remember_value(previous if isinstance(previous, bool) or previous is None else None, item)


def _status_from_memory(memory: Any) -> dict[str, Any]:
    if isinstance(memory, list):
        if any(value is False for value in memory):
            return {"worn": False, "confidence": 0.0, "evidence": "missing_class", "status": "Missing"}
        if memory and all(value is True for value in memory):
            return {"worn": True, "confidence": 0.0, "evidence": "worn_class", "status": "Worn"}
        return _not_visible()
    if memory is True:
        return {"worn": True, "confidence": 0.0, "evidence": "worn_class", "status": "Worn"}
    if memory is False:
        return {"worn": False, "confidence": 0.0, "evidence": "missing_class", "status": "Missing"}
    return _not_visible()


def stabilize_live_ppe(ppe: dict[str, Any], track_id: str = "primary") -> dict[str, Any]:
    """Remember gear for one tracked person.

    A worn or missing call sticks until a later visible frame contradicts it.
    A part that is out of frame leaves the memory unchanged and is not a miss.
    Gloves and shoes are complete only after each visible side has been seen worn.
    """
    if not ppe or not ppe.get("available"):
        return ppe
    if not ppe.get("person_seen"):
        return ppe

    items_in = ppe.get("items") or {}
    keys = tuple(items_in) or ppe_items()
    with _vote_lock:
        memory = _votes.setdefault(str(track_id), {})
        for key in keys:
            current = dict(items_in.get(key) or _empty_item())
            memory[key] = _remember_item(memory.get(key), current)

        items = {}
        for key in keys:
            item = _status_from_memory(memory.get(key))
            item["label"] = _item_label(key, items_in.get(key))
            items[key] = item

    summary, complete, missing = _summarize(items, True)
    stable = dict(ppe)
    stable["items"] = items
    stable["summary"] = summary
    stable["ppe_complete"] = complete
    stable["missing"] = missing
    for key in keys:
        stable[key] = items[key]["worn"]
    return stable


def wearings_for_fusion(ppe: dict[str, Any]) -> dict[str, bool] | None:
    """Return hard_hat/vest/… bools if the model made a decision; else None."""
    if not ppe.get("available") or ppe.get("source") not in {"vision_ppe"}:
        return None
    out: dict[str, bool] = {}
    for key in ppe.get("items") or ppe_items():
        worn = ppe.get(key)
        if worn is None:
            return None
        out[key] = bool(worn)
    return out
