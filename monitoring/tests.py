import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from monitoring.services.control_center import assemble, decide
from monitoring.services.distance_estimation import aim_hits, measure_person
from monitoring.services.ppe_inference import _decide_item, _summarize, assess_person, is_worker_box
from monitoring.services.recordings import log_line

_FUSION_OUT = {
    "label": "CAUTION",
    "label_idx": 1,
    "confidence": 0.71,
    "probabilities": {"NORMAL": 0.1, "CAUTION": 0.71, "DANGER": 0.19},
    "model_id": "fusion_test",
    "feature_cols": [],
}
_MEASURED_VISION = {
    "worker_detected": True,
    "confidence": 0.9,
    "worker_distance_est": 0.5,
    "distance_source": "vision",
    "worker_speed": 0.4,
    "approaching": True,
}
_VIBRATION = {"label": "NORMAL", "abnormal": False, "anomaly_score": 0.2}


def _pose(joints):
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in joints.items():
        points[index] = [float(point[0]), float(point[1]), float(point[2])]
    return points


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


class FusionDecisionTests(SimpleTestCase):
    def test_the_decision_is_the_fusion_label_as_it_comes_out(self):
        with patch("monitoring.services.control_center.predict_fusion", return_value=_FUSION_OUT) as predict:
            payload = assemble(
                {
                    **_MEASURED_VISION,
                    "ppe": {"available": True},
                    "ppe_complete": False,
                    "hazards": {"available": True, "fire_score": 0.9, "smoke_score": 0.1},
                    "falls_available": True,
                    "fall_detected": True,
                },
                _VIBRATION,
                machine={"label": "RUNNING", "running": True, "confidence": 0.9},
            )
        predict.assert_called_once()
        row = predict.call_args.args[0]
        self.assertEqual(row["worker_distance"], 0.5)
        self.assertEqual(row["inside_danger_zone"], 1.0)
        self.assertEqual(row["worker_speed"], 0.4)
        self.assertEqual(row["approaching_machine"], 1.0)
        self.assertEqual(row["vision_confidence"], 0.9)
        self.assertEqual(row["vibration_anomaly_score"], 0.2)
        self.assertEqual(row["machine_active"], 1.0)
        self.assertEqual(row["ppe_missing"], 1.0)
        self.assertEqual(row["fire_score"], 0.9)
        self.assertEqual(row["smoke_score"], 0.1)
        self.assertEqual(row["fall_detected"], 1.0)
        self.assertEqual(payload["fusion"]["label"], "CAUTION")
        self.assertEqual(payload["fusion"]["confidence"], 0.71)
        self.assertEqual(payload["fusion"]["reason"], "")

    def test_an_unmeasured_input_means_no_decision(self):
        cases = {
            "distance": ({**_MEASURED_VISION, "worker_distance_est": None}, _VIBRATION),
            "worker motion": ({**_MEASURED_VISION, "worker_speed": None, "approaching": None}, _VIBRATION),
            "vibration": (_MEASURED_VISION, None),
            "person": ({**_MEASURED_VISION, "confidence": None}, _VIBRATION),
        }
        for name, (vision, vibration) in cases.items():
            with patch("monitoring.services.control_center.predict_fusion") as predict:
                decision = decide(vision, vibration, machine_on=True)
            predict.assert_not_called()
            self.assertIsNone(decision["label"], name)
            self.assertIn(name, decision["missing"], name)
            self.assertIn(name, decision["reason"], name)

    def test_an_empty_frame_takes_the_absent_worker_values_the_model_was_trained_with(self):
        from monitoring.services.control_center import measured_inputs

        absent = {"worker_distance": 10.0, "inside_caution_zone": 0.0, "worker_speed": 0.0}
        empty = measured_inputs({"worker_detected": False}, _VIBRATION, machine_on=True, absent_worker=absent)
        self.assertEqual(empty["worker_detected"], 0.0)
        self.assertEqual(empty["worker_distance"], 10.0)
        self.assertEqual(empty["worker_speed"], 0.0)
        old_model = measured_inputs({"worker_detected": False}, _VIBRATION, machine_on=True, absent_worker={})
        self.assertNotIn("worker_distance", old_model)
        seen = measured_inputs(_MEASURED_VISION, _VIBRATION, machine_on=True, absent_worker=absent)
        self.assertEqual(seen["worker_distance"], 0.5)

    def test_ppe_missing_is_measured_only_when_the_ppe_model_ran_on_a_worker(self):
        from monitoring.services.control_center import measured_inputs

        missing = {**_MEASURED_VISION, "ppe": {"available": True}, "ppe_complete": False}
        uncertain = {**_MEASURED_VISION, "ppe": {"available": True}, "ppe_complete": None}
        no_model = {**_MEASURED_VISION, "ppe": {"available": False}, "ppe_complete": None}
        self.assertEqual(measured_inputs(missing, _VIBRATION, machine_on=True)["ppe_missing"], 1.0)
        self.assertEqual(measured_inputs(uncertain, _VIBRATION, machine_on=True)["ppe_missing"], 0.0)
        self.assertNotIn("ppe_missing", measured_inputs(no_model, _VIBRATION, machine_on=True))

    def test_no_decision_switches_the_buzzer_and_leds_off(self):
        from hardware.buzzer import Buzzer
        from hardware.leds import Leds

        self.assertEqual(Buzzer().apply_label(None), "off")
        self.assertEqual(Leds().apply_label(None), "off")
        self.assertEqual(Leds().apply_label("NORMAL"), "green")


class LogLineTests(SimpleTestCase):
    def test_only_a_real_distance_is_written_into_the_log(self):
        guessed = log_line(
            {
                "worker_detected": True,
                "summary": "Missing safety vest",
                "person_count": 1,
                "distance_m": 1.8,
                "distance_source": "heuristic",
            }
        )
        self.assertEqual(guessed, "Missing safety vest · 1 person")
        measured = log_line(
            {
                "worker_detected": True,
                "summary": "Complete",
                "person_count": 2,
                "distance_m": 1.4,
                "distance_source": "tof",
            },
            "NORMAL",
        )
        self.assertEqual(measured, "Complete · 2 people · 1.4 m sensor")
        self.assertEqual(log_line({"worker_detected": False}), "No person")


class DistanceTests(SimpleTestCase):
    def test_a_partial_body_still_gets_the_network_distance(self):
        estimate = measure_person(_pose({0: (150, 40, 1)}), (320, 480), box=[100, 20, 200, 200])
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


class VibrationModelTests(SimpleTestCase):
    def test_the_trained_model_labels_the_window(self):
        import numpy as np

        from monitoring.services import vibration_inference as vib

        class Model:
            classes_ = np.array([0, 1])

            def predict_proba(self, rows):
                return np.array([[0.3, 0.7]] * len(rows))

        window = np.zeros((100, 3), dtype=np.float32)
        window[:, 2] = 1.0
        meta = {
            "experiment_id": "vib_test",
            "class_names": ["NORMAL", "ABNORMAL"],
            "sample_rate_hz": 50.0,
            "window_size": 100,
            "feature_cols": list(vib.window_to_features(window, 50.0)),
        }
        with patch.object(vib, "get_model", return_value=Model()), patch.object(vib, "get_metadata", return_value=meta):
            self.assertIsNone(vib.predict_from_window(window[:60]))
            result = vib.predict_from_window(window)
        self.assertEqual(result["label"], "ABNORMAL")
        self.assertEqual(result["anomaly_score"], 0.7)
        self.assertEqual(result["model_id"], "vib_test")
        with patch.object(vib, "get_model", return_value=None):
            self.assertIsNone(vib.predict_from_window(window))

    def test_a_flat_window_has_no_nan_features(self):
        import math

        import numpy as np

        from monitoring.services.vibration_inference import window_to_features

        flat = np.zeros((100, 3), dtype=np.float32)
        flat[:, 2] = 1.0
        self.assertTrue(all(math.isfinite(value) for value in window_to_features(flat, 50.0).values()))


class ToFPresenceTests(SimpleTestCase):
    def test_tof_is_online_only_while_the_board_reports_it(self):
        from hardware.presence import get_presence
        from hardware.tof import get_tof

        presence = get_presence()
        presence.reset()
        try:
            self.assertFalse(get_tof().online())
            presence.note_line("# TOF raw=8190 timeout=0 status=4")
            self.assertTrue(get_tof().online())
            presence.note_line("# TOF not found model=0xFF")
            self.assertFalse(get_tof().online())
        finally:
            presence.reset()


class TrainingArchiveTests(TestCase):
    def setUp(self):
        from monitoring.models import Recording, RecordingMinute

        self.user = User.objects.create_user("operator", password="sams-review")
        self.recording = Recording.objects.create(title="yard.mp4", duration_s=70, fps=30, problem_count=1)
        self.minute = RecordingMinute.objects.create(
            recording=self.recording,
            minute=1,
            offset_s=60,
            level="CAUTION",
            summary="Missing safety vest",
            detail={"worker_detected": True, "distance_m": 1.4, "ppe": [{"key": "vest", "worn": False}]},
        )

    def test_keep_marks_the_minute_and_export_lists_it(self):
        self.client.login(username="operator", password="sams-review")
        response = self.client.post(
            reverse("review_label"),
            {"minute_id": self.minute.pk, "status": "kept", "next": "/mlops/"},
        )
        self.assertRedirects(response, "/mlops/")
        self.minute.refresh_from_db()
        self.assertEqual(self.minute.train_status, "kept")

        exported = self.client.get(reverse("review_export"))
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            labels = archive.read("labels.csv").decode()
        self.assertIn("yard.mp4", labels)
        self.assertIn("Missing safety vest", labels)
        self.assertIn("kept", labels)

    def test_home_archive_and_mlops_require_sign_in(self):
        self.client.login(username="operator", password="sams-review")
        for name in ("review_entrance", "review_archive", "review_alerts", "review_mlops"):
            page = self.client.get(reverse(name))
            self.assertEqual(page.status_code, 200, name)

    def test_home_overview_uses_archived_alerts(self):
        self.client.login(username="operator", password="sams-review")
        page = self.client.get(reverse("review_entrance"))
        self.assertContains(page, "Safety Overview")
        self.assertContains(page, "Missing safety vest")
        self.assertContains(page, "Phone camera")
        self.assertContains(page, "Not connected")
        self.assertContains(page, 'id="nav-toggle"')
        self.assertContains(page, "topbar-brand")
        week = self.client.get(reverse("review_entrance"), {"range": "7d"})
        self.assertContains(week, "per day")


class LiveOnlyTests(SimpleTestCase):
    def test_the_testing_console_and_simulate_routes_are_gone(self):
        for url in ("/system/", "/api/simulate/", "/api/vibration/predict/", "/api/vision/video/", "/api/hardware/mode/"):
            self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_machine_state_is_the_trained_models_label(self):
        import numpy as np

        from monitoring.services import machine_state_inference as machine_state
        from monitoring.services.vibration_inference import window_to_features

        class Model:
            classes_ = np.array([0, 1])

            def predict_proba(self, rows):
                return np.array([[0.8, 0.2]] * len(rows))

        window = np.zeros((100, 3), dtype=np.float32)
        window[:, 2] = 1.0
        meta = {
            "model_id": "machine_test",
            "class_names": ["STOPPED", "RUNNING"],
            "sample_rate_hz": 50.0,
            "window_size": 100,
            "feature_cols": [c for c in window_to_features(window, 50.0) if c.startswith("mag_")],
        }
        with patch.object(machine_state, "get_model", return_value=Model()), patch.object(machine_state, "get_metadata", return_value=meta):
            stopped = machine_state.predict_window(window)
            self.assertIsNone(machine_state.predict_window(None))
        self.assertEqual(stopped["label"], "STOPPED")
        self.assertFalse(stopped["running"])
        with patch.object(machine_state, "get_model", return_value=None):
            self.assertIsNone(machine_state.predict_window(window))
        with patch("monitoring.services.control_center.predict_fusion") as predict:
            decision = decide(_MEASURED_VISION, _VIBRATION, machine_on=None)
        predict.assert_not_called()
        self.assertIn("machine state", decision["missing"])


class InteractionLogTests(TestCase):
    def test_a_row_is_kept_only_when_the_decision_changes(self):
        from monitoring.models import InteractionEvent
        from monitoring.services.interaction_log import record_interaction, reset_interaction_signature

        reset_interaction_signature()
        payload = {
            "fusion": {"label": "CAUTION", "reason": "Unusual vibration."},
            "vision": {"people": [], "worker_distance_est": None, "distance_source": None},
            "features": {"machine_active": False, "approaching_machine": False},
            "vibration": {"abnormal": True, "anomaly_score": 0.6},
            "zone": "—",
        }
        self.assertIsNotNone(record_interaction(payload))
        self.assertIsNone(record_interaction(payload))
        changed = {
            **payload,
            "fusion": {"label": "NORMAL", "reason": ""},
            "vibration": {"abnormal": False, "anomaly_score": 0.02},
        }
        self.assertIsNotNone(record_interaction(changed))
        self.assertEqual(InteractionEvent.objects.count(), 2)
        first = InteractionEvent.objects.order_by("created_at", "pk").first()
        self.assertEqual(first.reason, "Unusual vibration.")
        self.assertFalse(first.machine_on)
        self.assertTrue(first.vibration_unusual)
        self.assertIsNone(record_interaction({**payload, "fusion": {"label": None, "reason": "No decision"}}))

    def test_alerts_page_lists_the_saved_interaction(self):
        from monitoring.models import InteractionEvent
        from monitoring.services.interaction_log import reset_interaction_signature

        reset_interaction_signature()
        User.objects.create_user("operator2", password="sams-review")
        InteractionEvent.objects.create(
            level="CAUTION",
            reason="Possible interaction with machine.",
            person_id="3",
            distance_m=0.6,
            distance_source="tof",
            machine_on=False,
        )
        self.client.login(username="operator2", password="sams-review")
        page = self.client.get(reverse("review_alerts"))
        self.assertContains(page, "Possible interaction with machine.")
        self.assertContains(page, "Live interactions")


_COLS = ["worker_distance", "vibration_anomaly_score"]


def _site_features():
    from monitoring.services.distance_estimation import FEATURE_NAMES

    return {name: 0.5 for name in FEATURE_NAMES}


def _live_payload(label):
    return {
        "fusion": {"label": label, "model_id": "fusion_test", "feature_cols": _COLS},
        "features": {"worker_distance": 0.6, "vibration_anomaly_score": 0.3},
    }


class TrialDataTests(TestCase):
    def setUp(self):
        from monitoring.services import trials

        self.trials = trials

    def test_without_a_trial_nothing_is_kept(self):
        from monitoring.models import TrainingSample

        self.assertIsNone(self.trials.record_live_samples(_live_payload("NORMAL"), {}))
        self.assertEqual(TrainingSample.objects.count(), 0)

    def test_a_measured_poll_is_kept_under_the_trials_true_label(self):
        from monitoring.models import TrainingSample

        trial = self.trials.start_trial(risk_label="DANGER", scenario="walk in")
        vision = {"distance_pairs": [{"features": _site_features(), "z0_m": 0.9, "distance_m": 0.62}]}
        status = self.trials.record_live_samples(_live_payload("NORMAL"), vision)
        self.assertEqual(status["code"], trial.code)
        self.assertEqual((status["fusion_rows"], status["distance_rows"]), (1, 1))
        row = TrainingSample.objects.get(kind="fusion")
        self.assertEqual(row.target, "DANGER")
        self.assertEqual(row.predicted, "NORMAL")
        self.assertEqual(row.features, {"worker_distance": 0.6, "vibration_anomaly_score": 0.3})
        self.assertEqual(TrainingSample.objects.get(kind="distance").target_value, 0.62)

    def test_a_poll_without_a_decision_gives_no_fusion_row(self):
        from monitoring.models import TrainingSample

        self.trials.start_trial(risk_label="CAUTION")
        self.trials.record_live_samples({"fusion": {"label": None}, "features": {}}, {})
        self.assertFalse(TrainingSample.objects.filter(kind="fusion").exists())

    def test_starting_a_trial_ends_the_running_one(self):
        first = self.trials.start_trial(risk_label="NORMAL")
        second = self.trials.start_trial(risk_label="CAUTION")
        first.refresh_from_db()
        self.assertIsNotNone(first.ended_at)
        self.assertEqual(self.trials.active_trial(), second)
        self.assertNotEqual(first.code, second.code)
        with self.assertRaises(ValueError):
            self.trials.start_trial(risk_label="SOMETIMES")

    def test_export_lays_the_rows_out_for_notebooks_03_and_04(self):
        from monitoring.services.distance_estimation import FEATURE_NAMES

        self.trials.start_trial(risk_label="DANGER", scenario="walk in")
        vision = {"distance_pairs": [{"features": _site_features(), "z0_m": 0.9, "distance_m": 0.62}]}
        self.trials.record_live_samples(_live_payload("CAUTION"), vision)
        with patch("monitoring.services.trials.get_feature_cols", return_value=_COLS):
            payload = self.trials.export_zip()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            fusion = archive.read(self.trials.FUSION_CSV).decode().splitlines()
            site = archive.read(self.trials.SITE_DISTANCE_CSV).decode().splitlines()
        self.assertEqual(
            fusion[0].split(","),
            [*self.trials.FUSION_HEADER, *_COLS, "label", "label_name", "predicted"],
        )
        self.assertTrue(fusion[1].startswith("trial-"))
        self.assertTrue(fusion[1].endswith(",0.6,0.3,2,DANGER,CAUTION"))
        self.assertEqual(site[0].split(","), ["trial_code", *FEATURE_NAMES, "z0_m", "distance_m"])
        self.assertTrue(site[1].endswith(",0.9,0.62"))

    def test_the_mlops_desk_starts_stops_and_deletes_trials(self):
        from monitoring.models import Trial

        User.objects.create_user("collector", password="sams-review")
        self.client.login(username="collector", password="sams-review")
        self.client.post(reverse("review_trial_start"), {"risk_label": "CAUTION", "scenario": "edge of zone"})
        trial = Trial.objects.get()
        page = self.client.get(reverse("review_mlops"))
        self.assertContains(page, f"REC {trial.code}")
        self.client.post(reverse("review_trial_stop"))
        trial.refresh_from_db()
        self.assertIsNotNone(trial.ended_at)
        self.client.post(reverse("review_trial_delete", args=[trial.pk]))
        self.assertFalse(Trial.objects.exists())
        self.assertEqual(self.client.get(reverse("review_trial_export")).status_code, 200)

    def test_stage_a_scene_uploads_a_video_with_only_the_machine_state(self):
        User.objects.create_user("stager", password="sams-review")
        self.client.login(username="stager", password="sams-review")
        for name in ("review_entrance", "review_archive"):
            page = self.client.get(reverse(name))
            self.assertContains(page, "Stage a scene")
            self.assertContains(page, 'action="%s" enctype="multipart/form-data"' % reverse("review_upload"))
            self.assertContains(page, 'name="machine_state" value="RUNNING"')
            self.assertContains(page, 'name="vibration" value="ABNORMAL"')
            self.assertNotContains(page, 'name="risk_label"')


def _jpeg(shade):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (shade, shade, shade)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _decided_payload(label):
    return {
        "fusion": {"label": label, "reason": "", "confidence": 0.8, "missing": []},
        "features": {"worker_distance": 0.6, "vibration_anomaly_score": 0.3},
        "vibration": {"label": "NORMAL", "anomaly_score": 0.3, "series": {"x": [0.0]}},
        "machine": {"label": "RUNNING"},
    }


_VISION_FRAME = {
    "label": "Person",
    "confidence": 0.9,
    "worker_detected": True,
    "person_count": 1,
    "people": [{"id": 1, "selected": True, "distance_m": 0.6, "distance_source": "vision", "ppe": {}}],
    "detections": [{"xyxy": [1, 2, 30, 40], "track_id": 1, "distance_m": 0.6, "distance_source": "vision"}],
    "frame_w": 64,
    "frame_h": 48,
}


class ArchiveRecordingTests(TestCase):
    def setUp(self):
        import tempfile

        from django.test import override_settings

        self.media = tempfile.mkdtemp()
        self.settings_override = override_settings(MEDIA_ROOT=self.media)
        self.settings_override.enable()

    def tearDown(self):
        import shutil

        self.settings_override.disable()
        shutil.rmtree(self.media, ignore_errors=True)

    def test_a_live_session_becomes_an_archive_recording_with_its_cards(self):
        from monitoring.models import Recording
        from monitoring.services.live_recorder import LiveRecorder, finalize

        recorder = LiveRecorder()
        recorder.capture("tok", _jpeg(10), _decided_payload("NORMAL"), _VISION_FRAME)
        recorder.capture("tok", _jpeg(10), _decided_payload("NORMAL"), _VISION_FRAME)
        recorder.capture("tok", _jpeg(200), _decided_payload("DANGER"), _VISION_FRAME)
        recording = Recording.objects.get()
        self.assertEqual(recording.source, "live")
        self.assertEqual(recording.render_status, "pending")
        cards = list(recording.minutes.order_by("minute"))
        self.assertEqual([card.level for card in cards], ["NORMAL", "DANGER"])
        self.assertEqual(cards[1].detail["features"], {"worker_distance": 0.6, "vibration_anomaly_score": 0.3})
        self.assertNotIn("series", cards[1].detail["vibration"])
        self.assertEqual(cards[1].detail["boxes"][0]["kind"], "person")
        self.assertTrue(cards[1].snapshot)

        finalize(recording.pk)
        recording.refresh_from_db()
        self.assertEqual(recording.render_status, "ready")
        self.assertTrue(recording.video)
        self.assertEqual(recording.problem_count, 1)

    def test_a_session_without_frames_is_dropped(self):
        from monitoring.models import Recording
        from monitoring.services.live_recorder import finalize

        recording = Recording.objects.create(title="Live", source="live")
        finalize(recording.pk)
        self.assertFalse(Recording.objects.exists())

    def test_declared_values_stand_in_for_the_board_and_say_so(self):
        from monitoring.services.video_analysis import clean_declared, declared_inputs

        declared = clean_declared("running", "abnormal")
        self.assertEqual(declared, {"machine_state": "RUNNING", "vibration": "ABNORMAL"})
        vibration, machine = declared_inputs(declared)
        self.assertEqual((vibration["anomaly_score"], vibration["declared"]), (1.0, True))
        self.assertTrue(machine["running"])
        self.assertEqual(clean_declared("", "sometimes"), {})
        self.assertEqual(declared_inputs({}), (None, None))

    def test_an_uploaded_clip_is_analysed_with_its_declared_inputs(self):
        from django.core.files.base import ContentFile

        from monitoring.models import Recording
        from monitoring.services import video_analysis
        from monitoring.services.recordings import encode_clip

        clip = Path(self.media) / "clip.mp4"
        encode_clip([(0.0, _jpeg(10)), (1.0, _jpeg(200))], clip)
        recording = Recording.objects.create(title="clip.mp4", source="upload", declared={"machine_state": "RUNNING", "vibration": "NORMAL"})
        recording.video.save("clip.mp4", ContentFile(clip.read_bytes()), save=True)

        class Tracking:
            now = 0.0

        seen = []

        def fake_predict(image, *, annotate, video):
            seen.append(video.now)
            return dict(_VISION_FRAME)

        with patch("monitoring.services.vision_inference.VideoTracking", Tracking), patch(
            "monitoring.services.vision_inference.predict_image", side_effect=fake_predict
        ), patch("monitoring.services.control_center.predict_fusion", return_value=_FUSION_OUT) as fused, patch(
            "monitoring.services.control_center.get_feature_cols", return_value=["machine_active", "vibration_state"]
        ):
            video_analysis.analyze_recording(recording.pk)
        row = fused.call_args.args[0]
        self.assertEqual(row, {"machine_active": 1.0, "vibration_state": 0.0})
        self.assertEqual([round(t, 2) for t in seen[:2]], [0.0, 0.6])  # 5 fps clip sampled every 0.5 s
        recording.refresh_from_db()
        self.assertEqual(recording.render_status, "ready")
        card = recording.minutes.first()
        self.assertEqual(card.level, "CAUTION")
        self.assertEqual(card.detail["declared"], {"machine_state": "RUNNING", "vibration": "NORMAL"})
        self.assertTrue(card.detail["machine"]["declared"])

    def test_a_kept_card_is_a_fusion_row_under_its_reviewed_label(self):
        from monitoring.models import Recording, RecordingMinute
        from monitoring.services import trials

        recording = Recording.objects.create(title="clip.mp4", source="upload")
        RecordingMinute.objects.create(
            recording=recording, minute=0, level="NORMAL", train_status="kept", true_label="DANGER",
            detail={"features": {"worker_distance": 0.6, "vibration_anomaly_score": 0.3}, "declared": {"vibration": "NORMAL"}},
        )
        RecordingMinute.objects.create(recording=recording, minute=1, level="CAUTION", train_status="pending", detail={})
        rows = trials._fusion_csv(_COLS).splitlines()
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1].startswith(f"rec-{recording.pk},clip.mp4,upload,,upload,vibration,0,"))
        self.assertTrue(rows[1].endswith(",0.6,0.3,2,DANGER,NORMAL"))

    def test_the_upload_form_keeps_the_clip_and_queues_it(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        from monitoring.models import Recording

        User.objects.create_user("uploader", password="sams-review")
        self.client.login(username="uploader", password="sams-review")
        with patch("monitoring.services.video_analysis.enqueue") as queued:
            response = self.client.post(
                reverse("review_upload"),
                {"video": SimpleUploadedFile("yard.mp4", b"0" * 200, content_type="video/mp4"), "machine_state": "STOPPED"},
            )
        recording = Recording.objects.get()
        self.assertRedirects(response, reverse("review_recording", args=[recording.pk]), fetch_redirect_response=False)
        queued.assert_called_once_with(recording.pk)
        self.assertEqual(recording.declared, {"machine_state": "STOPPED"})
        page = self.client.get(reverse("review_recording", args=[recording.pk]))
        self.assertContains(page, "Declared at upload, not measured: machine stopped.")


class SerialPortTests(SimpleTestCase):
    def test_an_explicit_port_is_used(self):
        import os

        from hardware.bringup import find_serial_port

        previous = os.environ.get("SAMS_HW_SERIAL_PORT")
        os.environ["SAMS_HW_SERIAL_PORT"] = "/dev/cu.usbserial-test"
        try:
            self.assertEqual(find_serial_port(), "/dev/cu.usbserial-test")
        finally:
            if previous is None:
                os.environ.pop("SAMS_HW_SERIAL_PORT", None)
            else:
                os.environ["SAMS_HW_SERIAL_PORT"] = previous


class HardwareHealthTests(TestCase):
    def _rows(self):
        from monitoring.review_views import _health

        return {row["label"]: row for row in _health()}

    def test_absent_sensors_stay_inactive(self):
        import os

        from hardware import get_stream
        from hardware.presence import get_presence

        stream = get_stream()
        previous = stream._source
        stream._source = None
        get_presence().reset()
        os.environ["SAMS_HW_TOF_M"] = "1.2"
        try:
            rows = self._rows()
        finally:
            stream._source = previous
            get_presence().reset()
            os.environ.pop("SAMS_HW_TOF_M", None)
        for name in ("Phone camera", "Accelerometer", "Distance sensor", "Buzzer"):
            self.assertFalse(rows[name]["ok"], name)
            self.assertEqual(rows[name]["status"], "Not connected")

    def test_only_a_sensor_the_board_reports_is_active(self):
        from hardware import get_stream
        from hardware.presence import get_presence

        class _Open:
            connected = True

        stream = get_stream()
        previous = stream._source
        stream._source = _Open()
        presence = get_presence()
        presence.reset()
        presence.note_sample()
        presence.note_line("# ACK B1")
        presence.note_line("# TOF not found")
        try:
            rows = self._rows()
        finally:
            stream._source = previous
            presence.reset()
        self.assertTrue(rows["Accelerometer"]["ok"])
        self.assertTrue(rows["Buzzer"]["ok"])
        self.assertFalse(rows["Distance sensor"]["ok"])
        self.assertFalse(rows["Phone camera"]["ok"])
        self.assertEqual(rows["Distance sensor"]["status"], "Not connected")



class MachineMotorTests(SimpleTestCase):
    def setUp(self):
        from hardware.motor import Motor

        self.sent: list[str] = []
        self.motor = Motor()
        patcher = patch("monitoring.views.get_motor", return_value=self.motor)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, level):
        return self.client.post("/api/hardware/motor/", {"level": level}, content_type="application/json")

    def _acking_writer(self, cmd):
        self.sent.append(cmd)
        self.motor.note_ack(f"# ACK {cmd}")

    def test_levels_send_serial_commands(self):
        self.motor.attach_writer(self._acking_writer)
        for level in ("low", "medium", "high", "intermittent", "off"):
            self.assertEqual(self._post(level).status_code, 200, level)
        self.assertEqual(self.sent, ["M120", "M180", "M255", "V1", "V0"])
        self.assertEqual(self.motor.status()["level"], "off")
        self.assertTrue(self.motor.status()["acked"])

    def test_old_firmware_shows_as_unacknowledged(self):
        self.motor.ACK_TIMEOUT_S = 0.05
        self.motor.attach_writer(self.sent.append)  # never answers, like firmware without M<duty>
        self.assertEqual(self._post("low").status_code, 200)
        self.assertFalse(self.motor.status()["acked"])

    def test_needs_the_board(self):
        res = self._post("high")
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.json()["motor"]["level"], "off")

    def test_rejects_unknown_level(self):
        self.motor.attach_writer(self.sent.append)
        self.assertEqual(self._post("warp").status_code, 400)
        self.assertEqual(self.sent, [])

    def test_stream_stop_turns_the_machine_off(self):
        self.motor.attach_writer(self.sent.append)
        self.motor.set_level("high")
        self.motor.detach()
        self.assertEqual(self.sent[-1], "V0")
        self.assertFalse(self.motor.status()["linked"])

    def test_vibration_feed_is_centred_per_axis(self):
        import numpy as np

        from hardware import get_stream

        win = np.zeros((100, 3), dtype=np.float32)
        win[:, 2] = 1.0
        win[::2, 0] = 0.1  # x swings 0 / 0.1 g
        with patch.object(get_stream(), "latest_window", return_value=win):
            data = self.client.get("/api/hardware/vibration/").json()
        self.assertTrue(data["ready"])
        self.assertAlmostEqual(data["series"]["z"][0], 0.0)
        self.assertAlmostEqual(data["rms_mg"]["x"], 50.0, places=0)
        self.assertEqual(data["rms_mg"]["z"], 0.0)
