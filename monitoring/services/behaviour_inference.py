"""Safe / unsafe workplace behaviour from one camera frame.

Weights and class names come from ``models/vision/safe_unsafe/``, written by
``notebooks/01_vision_safe_unsafe.ipynb``. The live service serves a
single-frame image model (ResNet18 fine-tune). The notebook's video-clip
winner is kept for research and is not loaded here.

Without those weights the result is ``available=False`` and the rest of the
pipeline keeps working.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .model_loader import LazyModel

logger = logging.getLogger("monitoring.behaviour")

CLASS_NAMES = [
    "Authorized Intervention",
    "Carrying Overload with Forklift",
    "Closed Panel Cover",
    "Opened Panel Cover",
    "Safe Carrying",
    "Safe Walkway",
    "Safe Walkway Violation",
    "Unauthorized Intervention",
]

DEFAULT_UNSAFE = {
    "Carrying Overload with Forklift",
    "Opened Panel Cover",
    "Safe Walkway Violation",
    "Unauthorized Intervention",
}


def _project_root() -> Path:
    try:
        from django.conf import settings

        if settings.configured:
            return Path(settings.BASE_DIR)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    custom = os.environ.get("SAMS_BEHAVIOUR_DIR", "").strip()
    if custom:
        return Path(custom)
    return _project_root() / "models" / "vision" / "safe_unsafe"


def weights_path() -> Path:
    custom = os.environ.get("SAMS_BEHAVIOUR_WEIGHTS", "").strip()
    if custom:
        return Path(custom)
    return model_dir() / "best.pt"


def metadata_path() -> Path:
    return model_dir() / "metadata.json"


def behaviour_enabled() -> bool:
    return os.environ.get("SAMS_VISION_BEHAVIOUR", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_spec() -> dict[str, Any]:
    path = metadata_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Behaviour metadata not found at {path}. "
            "Export it with notebooks/01_vision_safe_unsafe.ipynb"
        )
    spec = json.loads(path.read_text())
    for key in ("class_names", "unsafe_classes", "architecture", "num_classes", "img_size"):
        if key not in spec:
            raise ValueError(f"Behaviour metadata at {path} is missing {key}")
    return spec


def model_id() -> str:
    try:
        spec = load_spec()
        return str(spec.get("model_id") or spec.get("promoted_from") or spec.get("experiment_id") or "behaviour")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "behaviour"


class TinyCNN(nn.Module):
    """Matches the notebook baseline architecture."""

    def __init__(self, num_classes: int):
        super().__init__()
        blocks: list[nn.Module] = []
        for c_in, c_out, pool in ((3, 32, True), (32, 64, True), (64, 128, True), (128, 256, False)):
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            if pool:
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _build_architecture(spec: dict[str, Any]) -> nn.Module:
    """Rebuild the notebook network so ``state_dict`` keys match."""
    name = str(spec.get("architecture") or "").strip().lower().replace(" ", "")
    num_classes = int(spec["num_classes"])
    # ImageNet weights are not needed: we load the fine-tuned state dict.
    if name in {"resnet18", "resnet18(imagenet)"}:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"efficientnet-b0", "efficientnet_b0", "efficientnetb0", "efficientnet-b0(imagenet)"}:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name in {"mobilenetv3-small", "mobilenet_v3_small", "mobilenetv3small"}:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if name in {"convnext-tiny", "convnext_tiny", "convnexttiny"}:
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name in {"tinycnn"}:
        return TinyCNN(num_classes)
    if "r(2+1)d" in name or "r2plus1d" in name or "video" in name:
        raise ValueError(
            f"Behaviour checkpoint architecture {spec.get('architecture')!r} is a video model. "
            "Promote an image experiment (e.g. ResNet18) for live single-frame serving."
        )
    raise ValueError(f"Unsupported behaviour architecture: {spec.get('architecture')!r}")


def _eval_transform(spec: dict[str, Any]) -> transforms.Compose:
    size = int(spec.get("img_size") or 224)
    norm = spec.get("normalization") or {}
    mean = list(norm.get("mean") or [0.485, 0.456, 0.406])
    std = list(norm.get("std") or [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class BehaviourModel(LazyModel):
    """ResNet (or other image) checkpoint from the safe/unsafe notebook."""

    def __init__(self):
        super().__init__("behaviour", retry_after_error=True)
        self.spec: dict[str, Any] = {}
        self.transform: transforms.Compose | None = None

    def load(self) -> nn.Module:
        path = weights_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Behaviour weights not found at {path}. "
                "Copy the promoted model.pt from notebooks/01_vision_safe_unsafe.ipynb"
            )
        self.spec = load_spec()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise TypeError(f"{path} is not a notebook behaviour checkpoint (missing model_state_dict)")
        model = _build_architecture(self.spec)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.transform = _eval_transform(self.spec)
        return model

    def reset(self) -> None:
        super().reset()
        self.spec = {}
        self.transform = None


_loader = BehaviourModel()


def get_model() -> nn.Module | None:
    if not behaviour_enabled() or not weights_path().is_file() or not metadata_path().is_file():
        return None
    return _loader.get()


def reload_model() -> None:
    _loader.reset()


def get_status() -> dict[str, Any]:
    path = weights_path()
    return {
        "enabled": behaviour_enabled(),
        "weights": str(path) if path.is_file() else None,
        "available": behaviour_enabled() and path.is_file() and metadata_path().is_file() and _loader.error is None,
        "model_id": model_id(),
        "error": _loader.error,
    }


def _blank(*, available: bool = False) -> dict[str, Any]:
    return {
        "available": available,
        "model_id": model_id(),
        "unsafe": None,
        "label": None,
        "group": None,
        "confidence": None,
        "probabilities": None,
        "unsafe_score": None,
    }


@torch.inference_mode()
def classify_frame(image: Image.Image) -> dict[str, Any]:
    """Classify one RGB frame as an 8-class behaviour and SAFE/UNSAFE group."""
    if not behaviour_enabled():
        return _blank()
    if not weights_path().is_file() or not metadata_path().is_file():
        return _blank()
    model = get_model()
    if model is None or _loader.transform is None:
        return {**_blank(), "error": _loader.error}

    spec = _loader.spec or load_spec()
    class_names = list(spec.get("class_names") or CLASS_NAMES)
    unsafe_names = set(spec.get("unsafe_classes") or DEFAULT_UNSAFE)
    tensor = _loader.transform(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    label = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
    group = "UNSAFE" if label in unsafe_names else "SAFE"
    unsafe_score = float(sum(float(probs[i]) for i, name in enumerate(class_names) if name in unsafe_names))
    return {
        "available": True,
        "model_id": model_id(),
        "unsafe": group == "UNSAFE",
        "label": label,
        "group": group,
        "confidence": float(probs[idx]),
        "unsafe_score": unsafe_score,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
    }
