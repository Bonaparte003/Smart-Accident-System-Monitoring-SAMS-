"""Person, PPE, distance and motion: how each frame's detections are joined per worker."""

from unittest.mock import patch

from django.test import SimpleTestCase

from monitoring.services.distance_estimation import aim_hits, measure_person
from monitoring.services.ppe_inference import _decide_item, _summarize, assess_person, is_worker_box

from .fixtures import pose


class PpeDecisionTests(SimpleTestCase):
    def test_no_box_for_an_item_is_not_detected(self):
        item = _decide_item([], [])
        self.assertIsNone(item["worn"])
        self.assertEqual(item["status"], "Not detected")

    def test_the_stronger_box_decides(self):
        self.assertTrue(_decide_item([0.8], [0.3])["worn"])
        self.assertFalse(_decide_item([0.3], [0.8])["worn"])

    def test_complete_requires_every_item_worn(self):
        worn = {"worn": True, "confidence": 0.9, "evidence": "worn_class", "status": "Worn"}
        unknown = {"worn": None, "confidence": 0.0, "evidence": "none", "status": "Not detected"}
        missing = {"worn": False, "confidence": 0.7, "evidence": "missing_class", "status": "Missing"}
        items = {"hard_hat": dict(worn), "vest": dict(unknown), "gloves": dict(worn), "boots": dict(worn)}
        self.assertEqual(_summarize(items, True), ("Uncertain", None, []))
        items["vest"] = dict(missing)
        self.assertEqual(_summarize(items, True), ("Missing safety vest", False, ["Safety vest"]))
        items["vest"] = dict(worn)
        self.assertEqual(_summarize(items, True)[:2], ("Complete", True))

    def test_a_person_box_is_a_worker_and_a_vest_box_is_not(self):
        self.assertTrue(is_worker_box("person"))
        self.assertTrue(is_worker_box("Person"))
        self.assertFalse(is_worker_box("vest"))
        self.assertFalse(is_worker_box("helmet"))

    def test_every_model_box_on_the_person_counts(self):
        gear = [
            {"item": "hard_hat", "label": "helmet", "confidence": 0.9, "xyxy": [140, 64, 160, 76], "missing": False},
            {"item": "gloves", "label": "gloves", "confidence": 0.5, "xyxy": [100, 240, 120, 260], "missing": False},
            {"item": "gloves", "label": "no_gloves", "confidence": 0.8, "xyxy": [180, 240, 200, 260], "missing": True},
        ]
        result = assess_person(gear)
        self.assertTrue(result["hard_hat"])
        self.assertFalse(result["gloves"])
        self.assertIsNone(result["vest"])
        self.assertEqual(result["summary"], "Missing gloves")
        self.assertEqual(len(result["detections"]), 3)


class PpeThresholdTests(SimpleTestCase):
    def test_each_class_keeps_boxes_from_its_own_threshold(self):
        import numpy as np

        from monitoring.services import ppe_inference

        class Values:
            def __init__(self, data):
                self.data = np.array(data)

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        class Boxes:
            xyxy = Values([[0, 0, 10, 10]] * 3)
            conf = Values([0.10, 0.10, 0.30])
            cls = Values([0, 1, 1])
            id = None

            def __len__(self):
                return 3

        class Result:
            names = {0: "no_helmet", 1: "helmet"}
            boxes = Boxes()

        class Model:
            def predict(self, image, **kwargs):
                self.kwargs = kwargs
                return [Result()]

        spec = {
            "items": {"hard_hat": {"label": "Hard hat", "worn": ["helmet"], "missing": ["no_helmet"]}},
            "person": ["Person"],
            "conf": 0.25,
            "class_conf": {"no_helmet": 0.05, "helmet": 0.2},
            "imgsz": 640,
        }
        model = Model()
        with patch.object(ppe_inference, "load_spec", return_value=spec), patch.object(
            ppe_inference, "weights_available", return_value=True
        ), patch.object(ppe_inference, "spec_available", return_value=True):
            _workers, gear, _error = ppe_inference.detect_scene(None, model=model)
        self.assertEqual(model.kwargs["conf"], 0.05)
        self.assertEqual([(det["label"], det["confidence"]) for det in gear], [("no_helmet", 0.1), ("helmet", 0.3)])


class AssignmentTests(SimpleTestCase):
    def test_a_box_belongs_to_the_person_holding_its_centre(self):
        from monitoring.services.vision_inference import _assign

        people = [[0, 0, 100, 200], [80, 0, 140, 100]]
        boxes = [
            {"xyxy": [10, 10, 30, 30]},
            {"xyxy": [85, 10, 95, 30]},
            {"xyxy": [300, 300, 320, 320]},
        ]
        owned, unowned = _assign(people, boxes)
        self.assertEqual(owned[0], [boxes[0]])
        self.assertEqual(owned[1], [boxes[1]])
        self.assertEqual(unowned, [boxes[2]])


class PoseDrawTests(SimpleTestCase):
    def test_overlay_config_comes_from_the_pose_spec(self):
        from monitoring.services import vision_inference as vision

        vision._pose.spec = {
            "skeleton": [[5, 7], [7, 9]],
            "keypoint_names": ["a", "b"],
            "operating_point": {"kpt_conf": 0.4},
        }
        config = vision.pose_overlay_config()
        self.assertEqual(config["skeleton"], [[5, 7], [7, 9]])
        self.assertEqual(config["kpt_conf"], 0.4)
        self.assertEqual(config["keypoint_names"], ["a", "b"])

    def test_joints_are_painted_from_model_skeleton_only(self):
        from PIL import Image, ImageDraw

        from monitoring.services import vision_inference as vision

        vision._pose.spec = {
            "skeleton": [[0, 1]],
            "operating_point": {"kpt_conf": 0.3},
        }
        image = Image.new("RGB", (40, 40), "black")
        draw = ImageDraw.Draw(image)
        vision._draw_pose(
            draw,
            [[5, 5, 0.9], [20, 20, 0.9], [30, 30, 0.1]],
            bone_color="#00ff00",
        )
        # High-confidence joints and their bone leave green pixels; the low one does not.
        self.assertEqual(image.getpixel((5, 5))[1], 255)
        self.assertEqual(image.getpixel((20, 20))[1], 255)
        self.assertEqual(image.getpixel((30, 30)), (0, 0, 0))

    def test_no_skeleton_still_paints_visible_joints(self):
        from PIL import Image, ImageDraw

        from monitoring.services import vision_inference as vision

        vision._pose.spec = {"operating_point": {"kpt_conf": 0.5}}
        image = Image.new("RGB", (20, 20), "black")
        draw = ImageDraw.Draw(image)
        vision._draw_pose(draw, [[8, 8, 0.9]], bone_color="#0000ff")
        self.assertNotEqual(image.getpixel((8, 8)), (0, 0, 0))


class DistanceTests(SimpleTestCase):
    def test_a_partial_body_still_gets_the_network_distance(self):
        estimate = measure_person(pose({0: (150, 40, 1)}), (320, 480), box=[100, 20, 200, 200])
        self.assertEqual(estimate.source, "vision")
        self.assertTrue(str(estimate.method).startswith("residual_mlp:"))
        self.assertGreater(estimate.distance_m, 0)

    def test_the_sensor_is_used_for_the_person_it_points_at_within_its_range(self):
        from monitoring.services.distance_estimation import load_calib

        box = [100, 20, 220, 470]
        aimed = measure_person(None, (320, 480), box=box, tof_m=1.2, sensor_aimed=True)
        self.assertEqual(aimed.source, "tof")
        self.assertEqual(aimed.distance_m, 1.2)
        self.assertIsNotNone(aimed.body_m)
        beyond = measure_person(
            None, (320, 480), box=box, tof_m=float(load_calib()["tof_near_m"]) + 1.0, sensor_aimed=True
        )
        self.assertEqual(beyond.source, "vision")
        off_beam = measure_person(None, (320, 480), box=box, tof_m=1.2, sensor_aimed=False)
        self.assertEqual(off_beam.source, "vision")

    def test_the_beam_is_the_image_centre(self):
        self.assertFalse(aim_hits([0, 0, 10, 10], (100, 100)))
        self.assertTrue(aim_hits([40, 40, 60, 80], (100, 100)))


class MotionTests(SimpleTestCase):
    def test_motion_is_the_slope_of_measured_distances(self):
        from monitoring.services.person_tracker import MotionLog

        log = MotionLog(history_s=1.5)
        log.record_distance(1, 3.0, now=0.0)
        self.assertIsNone(log.motion(1)["speed_mps"])
        log.record_distance(1, 2.5, now=0.5)
        motion = log.motion(1)
        self.assertAlmostEqual(motion["speed_mps"], 1.0)
        self.assertAlmostEqual(motion["approach_mps"], 1.0)
        self.assertTrue(motion["approaching"])
