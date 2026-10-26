"""The fusion decision: inputs it receives, and no decision when an input is unmeasured."""

from unittest.mock import patch

from django.test import SimpleTestCase

from monitoring.services.control_center import assemble, decide
from monitoring.services.recordings import log_line

from .fixtures import FUSION_OUT, MEASURED_VISION, VIBRATION


class FusionDecisionTests(SimpleTestCase):
    def test_the_decision_is_the_fusion_label_as_it_comes_out(self):
        with patch("monitoring.services.control_center.predict_fusion", return_value=FUSION_OUT) as predict:
            payload = assemble(
                {
                    **MEASURED_VISION,
                    "ppe": {"available": True},
                    "ppe_complete": False,
                    "hazards": {"available": True, "fire_score": 0.9, "smoke_score": 0.1},
                    "falls_available": True,
                    "fall_detected": True,
                },
                VIBRATION,
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
            "distance": ({**MEASURED_VISION, "worker_distance_est": None}, VIBRATION),
            "worker motion": ({**MEASURED_VISION, "worker_speed": None, "approaching": None}, VIBRATION),
            "vibration": (MEASURED_VISION, None),
            "person": ({**MEASURED_VISION, "confidence": None}, VIBRATION),
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
        empty = measured_inputs({"worker_detected": False}, VIBRATION, machine_on=True, absent_worker=absent)
        self.assertEqual(empty["worker_detected"], 0.0)
        self.assertEqual(empty["worker_distance"], 10.0)
        self.assertEqual(empty["worker_speed"], 0.0)
        old_model = measured_inputs({"worker_detected": False}, VIBRATION, machine_on=True, absent_worker={})
        self.assertNotIn("worker_distance", old_model)
        seen = measured_inputs(MEASURED_VISION, VIBRATION, machine_on=True, absent_worker=absent)
        self.assertEqual(seen["worker_distance"], 0.5)

    def test_ppe_missing_is_measured_only_when_the_ppe_model_ran_on_a_worker(self):
        from monitoring.services.control_center import measured_inputs

        missing = {**MEASURED_VISION, "ppe": {"available": True}, "ppe_complete": False}
        uncertain = {**MEASURED_VISION, "ppe": {"available": True}, "ppe_complete": None}
        no_model = {**MEASURED_VISION, "ppe": {"available": False}, "ppe_complete": None}
        self.assertEqual(measured_inputs(missing, VIBRATION, machine_on=True)["ppe_missing"], 1.0)
        self.assertEqual(measured_inputs(uncertain, VIBRATION, machine_on=True)["ppe_missing"], 0.0)
        self.assertNotIn("ppe_missing", measured_inputs(no_model, VIBRATION, machine_on=True))

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
