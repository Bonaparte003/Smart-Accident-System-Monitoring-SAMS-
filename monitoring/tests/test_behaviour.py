"""Safe/unsafe behaviour classifier: architecture rebuild and unsafe flag merge."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from PIL import Image

from monitoring.services import behaviour_inference as behaviour
from monitoring.services.vision_inference import _unsafe_flag


class BehaviourArchitectureTests(SimpleTestCase):
    def test_resnet18_rebuilds_with_eight_outputs(self):
        model = behaviour._build_architecture(
            {"architecture": "ResNet18", "num_classes": 8, "img_size": 224}
        )
        self.assertEqual(model.fc.out_features, 8)

    def test_video_architecture_is_rejected_for_live_frames(self):
        with self.assertRaises(ValueError) as ctx:
            behaviour._build_architecture(
                {"architecture": "R(2+1)D-18 (Kinetics-400)", "num_classes": 8, "img_size": 224}
            )
        self.assertIn("video model", str(ctx.exception).lower())


class BehaviourUnsafeFlagTests(SimpleTestCase):
    def test_behaviour_unsafe_or_missing_ppe(self):
        self.assertTrue(
            _unsafe_flag({"ppe_complete": True}, {"available": True, "unsafe": True})
        )
        self.assertTrue(
            _unsafe_flag({"ppe_complete": False}, {"available": True, "unsafe": False})
        )
        self.assertFalse(
            _unsafe_flag({"ppe_complete": True}, {"available": True, "unsafe": False})
        )

    def test_falls_back_to_ppe_when_behaviour_unavailable(self):
        self.assertTrue(_unsafe_flag({"ppe_complete": False}, {"available": False}))
        self.assertFalse(_unsafe_flag({"ppe_complete": True}, {"available": False}))
        self.assertFalse(_unsafe_flag({"ppe_complete": None}, {"available": False}))


class BehaviourClassifyTests(SimpleTestCase):
    def test_missing_weights_report_unavailable(self):
        with patch.object(behaviour, "weights_path", return_value=MagicMock(is_file=lambda: False)):
            result = behaviour.classify_frame(Image.new("RGB", (64, 64), color="gray"))
        self.assertFalse(result["available"])
        self.assertIsNone(result["unsafe"])

    def test_classify_frame_uses_loaded_model(self):
        import torch

        fake = MagicMock()
        fake.return_value = torch.zeros(1, 8)
        fake.return_value[0, 7] = 5.0  # Unauthorized Intervention -> UNSAFE

        spec = {
            "architecture": "ResNet18",
            "num_classes": 8,
            "img_size": 224,
            "class_names": behaviour.CLASS_NAMES,
            "unsafe_classes": sorted(behaviour.DEFAULT_UNSAFE),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "model_id": "test_behaviour",
        }
        with (
            patch.object(behaviour, "behaviour_enabled", return_value=True),
            patch.object(behaviour, "weights_path", return_value=MagicMock(is_file=lambda: True)),
            patch.object(behaviour, "metadata_path", return_value=MagicMock(is_file=lambda: True)),
            patch.object(behaviour, "get_model", return_value=fake),
            patch.object(behaviour, "load_spec", return_value=spec),
            patch.object(behaviour, "model_id", return_value="test_behaviour"),
        ):
            behaviour._loader.spec = spec
            behaviour._loader.transform = behaviour._eval_transform(spec)
            behaviour._loader.error = None
            result = behaviour.classify_frame(Image.new("RGB", (224, 224), color="gray"))

        self.assertTrue(result["available"])
        self.assertTrue(result["unsafe"])
        self.assertEqual(result["group"], "UNSAFE")
        self.assertEqual(result["label"], "Unauthorized Intervention")
