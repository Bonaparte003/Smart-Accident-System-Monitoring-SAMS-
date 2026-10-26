"""The shared lazy loader: one load across threads, remembered errors, reload."""

import json
import tempfile
import threading
from pathlib import Path

import joblib
from django.test import SimpleTestCase

from monitoring.services.model_loader import JoblibModel, LazyModel


class CountingModel(LazyModel):
    def __init__(self, *, fail: bool = False, retry_after_error: bool = True):
        super().__init__("counting", retry_after_error=retry_after_error)
        self.fail = fail
        self.loads = 0

    def load(self):
        self.loads += 1
        if self.fail:
            raise FileNotFoundError("no weights")
        return object()


class LazyModelTests(SimpleTestCase):
    def test_concurrent_requests_load_the_model_once(self):
        loader = CountingModel()
        results = []
        threads = [threading.Thread(target=lambda: results.append(loader.get())) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(loader.loads, 1)
        self.assertEqual(len({id(model) for model in results}), 1)

    def test_a_failed_load_is_reported_not_raised(self):
        loader = CountingModel(fail=True)
        self.assertIsNone(loader.get())
        self.assertEqual(loader.error, "no weights")

    def test_a_failed_load_is_retried_only_when_allowed(self):
        retrying = CountingModel(fail=True, retry_after_error=True)
        retrying.get()
        retrying.get()
        self.assertEqual(retrying.loads, 2)

        cached = CountingModel(fail=True, retry_after_error=False)
        cached.get()
        cached.get()
        self.assertEqual(cached.loads, 1)

    def test_raise_errors_raises_the_load_error(self):
        with self.assertRaises(FileNotFoundError):
            CountingModel(fail=True).get(raise_errors=True)
        cached = CountingModel(fail=True, retry_after_error=False)
        cached.get()
        with self.assertRaisesMessage(RuntimeError, "no weights"):
            cached.get(raise_errors=True)

    def test_reset_loads_new_weights_on_the_next_call(self):
        loader = CountingModel()
        first = loader.get()
        loader.reset()
        self.assertIsNot(loader.get(), first)
        self.assertEqual(loader.loads, 2)


class Classifier:
    def predict_proba(self, rows):
        return [[1.0]]


class JoblibModelTests(SimpleTestCase):
    def folder(self, model) -> Path:
        tmp = Path(tempfile.mkdtemp())
        joblib.dump(model, tmp / "model.joblib")
        (tmp / "metadata.json").write_text(json.dumps({"class_names": ["A"]}))
        return tmp

    def test_metadata_is_read_without_loading_the_model(self):
        loader = JoblibModel("test", lambda: self.folder(Classifier()))
        self.assertEqual(loader.read_metadata()["class_names"], ["A"])
        self.assertIsNone(loader.cached)

    def test_a_model_without_probabilities_is_rejected_when_they_are_required(self):
        folder = self.folder({"not": "a classifier"})
        self.assertIsNone(JoblibModel("test", lambda: folder, require_proba=True).get())
        self.assertIsNotNone(JoblibModel("test", lambda: folder).get())

    def test_loading_also_reads_the_metadata(self):
        loader = JoblibModel("test", lambda: self.folder(Classifier()))
        self.assertIsInstance(loader.get(), Classifier)
        self.assertEqual(loader.metadata["class_names"], ["A"])
