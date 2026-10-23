"""Lazy, thread-safe model loading shared by every model service.

Each service owns one ``LazyModel``. The first ``get`` loads the model under a
lock (double-checked, so concurrent requests load it once), later calls return
the cached object, and ``reset`` drops it so the next call loads new weights.
Subclasses only say how a model is read from disk.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("monitoring.models")


class LazyModel(ABC):
    """One model, loaded on first use and shared across threads.

    A failed load is kept in ``error``. With ``retry_after_error`` the next
    ``get`` tries again (weights may have been copied in since); without it,
    the model stays unavailable until ``reset``.
    """

    def __init__(self, name: str, *, retry_after_error: bool = True):
        self.name = name
        self.retry_after_error = retry_after_error
        self.error: str | None = None
        self._model: Any = None
        self._lock = threading.Lock()

    @abstractmethod
    def load(self) -> Any:
        """Read the model from disk. Raises when it cannot be loaded."""

    @property
    def cached(self) -> Any:
        """The model if it is already loaded, without loading it."""
        return self._model

    def _should_load(self) -> bool:
        return self._model is None and (self.error is None or self.retry_after_error)

    def get(self, *, raise_errors: bool = False) -> Any:
        """The loaded model, or None when it cannot be loaded (or raise, with ``raise_errors``)."""
        if self._should_load():
            with self._lock:
                if self._should_load():
                    try:
                        self._model = self.load()
                        self.error = None
                        logger.info("Loaded %s model", self.name)
                    except Exception as exc:  # noqa: BLE001
                        self.error = str(exc)
                        if raise_errors:
                            raise
                        logger.warning("%s model unavailable: %s", self.name, exc)
        if self._model is None and raise_errors and self.error:
            raise RuntimeError(self.error)
        return self._model

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self.error = None


class JoblibModel(LazyModel):
    """A scikit-learn ``model.joblib`` and the ``metadata.json`` next to it."""

    def __init__(
        self,
        name: str,
        folder: Callable[[], Path],
        *,
        require_proba: bool = False,
        retry_after_error: bool = False,
    ):
        super().__init__(name, retry_after_error=retry_after_error)
        self._folder = folder
        self.require_proba = require_proba
        self.metadata: dict[str, Any] = {}

    def load(self) -> Any:
        import joblib

        folder = self._folder()
        model = joblib.load(folder / "model.joblib")
        if self.require_proba and not hasattr(model, "predict_proba"):
            raise TypeError(f"{folder / 'model.joblib'} has no predict_proba; retrain it with its notebook")
        self.read_metadata()
        return model

    def read_metadata(self) -> dict[str, Any]:
        """The model's metadata.json, read once, without loading the model itself."""
        if not self.metadata:
            self.metadata = json.loads((self._folder() / "metadata.json").read_text())
        return self.metadata

    def reset(self) -> None:
        super().reset()
        self.metadata = {}


class YoloModel(LazyModel):
    """An Ultralytics YOLO checkpoint (``best.pt``)."""

    def __init__(self, name: str, weights: Callable[[], Path]):
        super().__init__(name, retry_after_error=True)
        self._weights = weights

    def load(self) -> Any:
        path = self._weights()
        if not path.is_file():
            raise FileNotFoundError(f"{self.name} weights not found at {path}")
        from ultralytics import YOLO

        return YOLO(str(path))
