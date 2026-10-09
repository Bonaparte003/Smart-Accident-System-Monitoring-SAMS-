import io
import zipfile

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from monitoring.services.control_center import assemble, classify
from monitoring.services.distance_estimation import aim_hits, measure_person, stature_px
from monitoring.services.recordings import active_log_index, log_line, stored_boxes
from monitoring.services.ppe_inference import (
    _decide_item,
    _summarize,
    assess_person,
    is_worker_box,
    reset_live_votes,
    stabilize_live_ppe,
    worker_has_head_or_shoulders,
)


def _pose(joints):
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in joints.items():
        points[index] = [float(point[0]), float(point[1]), float(point[2])]
    return points


class PpeDecisionTests(SimpleTestCase):
    def test_no_detection_stays_unknown(self):
        item = _decide_item([], [], person_seen=True)
        self.assertIsNone(item["worn"])
        self.assertEqual(item["evidence"], "uncertain")

    def test_no_detection_without_a_person_stays_unknown(self):
        item = _decide_item([], [], person_seen=False)
        self.assertIsNone(item["worn"])

    def test_complete_requires_every_item_worn(self):
        worn = {"worn": True, "confidence": 0.9, "evidence": "worn_class", "status": "Worn"}
        unknown = {"worn": None, "confidence": 0.0, "evidence": "uncertain", "status": "—"}
        missing = {"worn": False, "confidence": 0.0, "evidence": "absent", "status": "Missing"}
        items = {
            "hard_hat": dict(worn),
            "vest": dict(unknown),
            "gloves": dict(worn),
            "boots": dict(worn),
        }
        summary, complete, missing_labels = _summarize(items, True)
        self.assertEqual(summary, "Uncertain")
        self.assertIsNone(complete)
        self.assertEqual(missing_labels, [])

        items["vest"] = dict(missing)
        summary, complete, missing_labels = _summarize(items, True)
        self.assertEqual(summary, "Missing safety vest")
        self.assertFalse(complete)
        self.assertEqual(missing_labels, ["Safety vest"])

        items["vest"] = dict(worn)
        summary, complete, _missing_labels = _summarize(items, True)
        self.assertEqual(summary, "Complete")
        self.assertTrue(complete)

    def test_a_person_box_is_a_worker_and_a_vest_box_is_not(self):
        self.assertTrue(is_worker_box("person"))
        self.assertTrue(is_worker_box("Person"))
        self.assertFalse(is_worker_box("worker"))
        self.assertFalse(is_worker_box("vest"))
        self.assertFalse(is_worker_box("helmet"))

    def test_a_limb_without_a_head_or_shoulders_is_not_a_worker(self):
        self.assertFalse(worker_has_head_or_shoulders(_pose({15: (40, 400, 0.9), 16: (70, 400, 0.9)})))
        self.assertFalse(worker_has_head_or_shoulders(_pose({11: (40, 200, 0.9), 12: (80, 200, 0.9)})))
        self.assertTrue(worker_has_head_or_shoulders(_pose({5: (40, 80, 0.9)})))
        self.assertTrue(worker_has_head_or_shoulders(_pose({0: (60, 30, 0.9)})))
        self.assertTrue(worker_has_head_or_shoulders(_pose({5: (40, 80, 0.9), 11: (40, 200, 0.4)})))

    def test_live_vote_is_kept_per_person(self):
        reset_live_votes()
        frame = {
            "available": True,
            "person_seen": True,
            "items": {
                "hard_hat": {"worn": True, "confidence": 0.8, "evidence": "worn_class", "status": "Worn", "label": "Hard hat"},
                "vest": {"worn": True, "confidence": 0.8, "evidence": "worn_class", "status": "Worn", "label": "Safety vest"},
                "gloves": {"worn": False, "confidence": 0.6, "evidence": "missing_class", "status": "Missing", "label": "Gloves"},
                "boots": {"worn": True, "confidence": 0.7, "evidence": "worn_class", "status": "Worn", "label": "Safety shoes"},
            },
        }
        first = stabilize_live_ppe(frame, track_id="1")
        self.assertTrue(first["hard_hat"])
        self.assertFalse(first["gloves"])
        self.assertEqual(first["summary"], "Missing gloves")
        other = stabilize_live_ppe(
            {
                **frame,
                "items": {
                    **frame["items"],
                    "gloves": {"worn": True, "confidence": 0.8, "evidence": "worn_class", "status": "Worn", "label": "Gloves"},
                },
            },
            track_id="2",
        )
        self.assertTrue(other["gloves"])
        still = stabilize_live_ppe(frame, track_id="1")
        self.assertFalse(still["gloves"])
        self.assertFalse(still["ppe_complete"])
        reset_live_votes()

    def test_hidden_gear_is_filled_in_on_a_later_frame(self):
        reset_live_votes()
        hidden_boots = {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Safety shoes", "sides": [None, None]}
        worn = {"worn": True, "confidence": 0.8, "evidence": "worn_class", "status": "Worn"}
        first = stabilize_live_ppe(
            {
                "available": True,
                "person_seen": True,
                "items": {
                    "hard_hat": {**worn, "label": "Hard hat"},
                    "vest": {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Safety vest"},
                    "gloves": {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Gloves", "sides": [None, None]},
                    "boots": hidden_boots,
                },
            },
            track_id="face",
        )
        self.assertIsNone(first["boots"])
        self.assertNotIn("Safety shoes", first["missing"])
        self.assertEqual(first["items"]["boots"]["status"], "Not visible")
        second = stabilize_live_ppe(
            {
                "available": True,
                "person_seen": True,
                "items": {
                    "hard_hat": {**worn, "label": "Hard hat"},
                    "vest": {**worn, "label": "Safety vest"},
                    "gloves": {
                        "worn": True,
                        "confidence": 0.8,
                        "evidence": "worn_class",
                        "status": "Worn",
                        "label": "Gloves",
                        "sides": [{**worn}, {**worn}],
                    },
                    "boots": {
                        "worn": True,
                        "confidence": 0.8,
                        "evidence": "worn_class",
                        "status": "Worn",
                        "label": "Safety shoes",
                        "sides": [{**worn}, {**worn}],
                    },
                },
            },
            track_id="face",
        )
        self.assertTrue(second["boots"])
        self.assertTrue(second["gloves"])
        self.assertTrue(second["ppe_complete"])
        hidden_again = stabilize_live_ppe(
            {
                "available": True,
                "person_seen": True,
                "items": {
                    "hard_hat": {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Hard hat"},
                    "vest": {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Safety vest"},
                    "gloves": {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible", "label": "Gloves", "sides": [None, None]},
                    "boots": hidden_boots,
                },
            },
            track_id="face",
        )
        self.assertTrue(hidden_again["boots"])
        self.assertTrue(hidden_again["ppe_complete"])
        reset_live_votes()

    def test_one_glove_then_the_other_completes_the_pair(self):
        reset_live_votes()
        worn = {"worn": True, "confidence": 0.8, "evidence": "worn_class", "status": "Worn"}
        hidden = {"worn": None, "confidence": 0.0, "evidence": "not_visible", "status": "Not visible"}

        def frame(sides):
            return stabilize_live_ppe(
                {
                    "available": True,
                    "person_seen": True,
                    "items": {
                        "hard_hat": {**worn, "label": "Hard hat"},
                        "vest": {**worn, "label": "Safety vest"},
                        "gloves": {"label": "Gloves", "sides": sides, "worn": None, "evidence": "not_visible", "status": "Not visible", "confidence": 0.0},
                        "boots": {**worn, "label": "Safety shoes"},
                    },
                },
                track_id="hands",
            )

        first = frame([{**worn}, None])
        self.assertIsNone(first["gloves"])
        self.assertEqual(first["items"]["gloves"]["status"], "Not visible")
        second = frame([None, {**worn}])
        self.assertTrue(second["gloves"])
        self.assertEqual(second["items"]["gloves"]["status"], "Worn")
        reset_live_votes()

    def test_a_face_alone_does_not_flag_missing_boots(self):
        pose = _pose({0: (150, 80, 1), 1: (140, 78, 1), 2: (160, 78, 1)})
        result = assess_person(pose, [120, 40, 180, 120], [])
        self.assertEqual(result["items"]["boots"]["status"], "Not visible")
        self.assertIsNone(result["boots"])
        self.assertNotIn("Safety shoes", result["missing"])
        self.assertEqual(result["items"]["vest"]["status"], "Not visible")

    def test_gear_counts_only_inside_its_body_region(self):
        pose = _pose(
            {
                0: (150, 82, 1),
                1: (140, 78, 1),
                2: (160, 78, 1),
                5: (120, 140, 1),
                6: (180, 140, 1),
                9: (110, 250, 1),
                10: (190, 250, 1),
                11: (125, 230, 1),
                12: (175, 230, 1),
                15: (130, 380, 1),
                16: (170, 380, 1),
            }
        )
        gear = [
            {"item": "hard_hat", "label": "helmet", "confidence": 0.9, "xyxy": [140, 64, 160, 76], "missing": False},
            {"item": "hard_hat", "label": "helmet", "confidence": 0.9, "xyxy": [140, 370, 160, 390], "missing": False},
            {"item": "gloves", "label": "gloves", "confidence": 0.8, "xyxy": [100, 240, 120, 260], "missing": False},
            {"item": "gloves", "label": "no_gloves", "confidence": 0.8, "xyxy": [180, 240, 200, 260], "missing": True},
        ]
        result = assess_person(pose, [100, 50, 200, 420], gear)
        self.assertTrue(result["hard_hat"])
        self.assertIsNone(result["vest"])
        self.assertEqual(result["items"]["vest"]["evidence"], "uncertain")
        self.assertFalse(result["gloves"])
        self.assertEqual(result["items"]["boots"]["status"], "—")
        labels = {det["label"] for det in result["detections"]}
        self.assertIn("helmet", labels)
        self.assertNotIn("no_boots", labels)
        self.assertEqual(sum(1 for det in result["detections"] if det["label"] == "helmet"), 1)

    def test_a_glove_past_the_wrist_still_counts(self):
        pose = _pose({7: (110, 160, 1), 9: (110, 230, 1)})
        gear = [
            {"item": "gloves", "label": "gloves", "confidence": 0.8, "xyxy": [90, 270, 140, 330], "missing": False},
        ]
        result = assess_person(pose, [70, 80, 180, 400], gear)
        self.assertTrue(result["items"]["gloves"]["sides"][0]["worn"])
        self.assertIsNone(result["gloves"])

    def test_a_glove_below_the_model_confidence_does_not_count(self):
        pose = _pose({9: (110, 250, 1), 5: (120, 140, 1), 11: (125, 230, 1)})
        gear = [
            {"item": "gloves", "label": "no_gloves", "confidence": 0.2, "xyxy": [90, 230, 140, 280], "missing": True},
        ]
        result = assess_person(pose, [80, 40, 180, 420], gear)
        self.assertIsNone(result["items"]["gloves"]["sides"][0]["worn"])

    def test_hidden_hand_is_not_reported_as_missing(self):
        pose = _pose({9: (110, 250, 1), 5: (120, 140, 1), 11: (125, 230, 1)})
        gear = [
            {"item": "gloves", "label": "gloves", "confidence": 0.8, "xyxy": [100, 240, 120, 260], "missing": False},
        ]
        result = assess_person(pose, [100, 50, 200, 420], gear)
        self.assertIsNone(result["gloves"])
        self.assertEqual(result["items"]["gloves"]["status"], "Not visible")


class AlertRuleTests(SimpleTestCase):
    def test_a_network_distance_is_kept_and_the_model_label_is_the_decision(self):
        from unittest.mock import patch

        model = {
            "label": "NORMAL",
            "label_idx": 0,
            "confidence": 0.62,
            "probabilities": {"NORMAL": 0.62, "CAUTION": 0.3, "DANGER": 0.08},
            "model_id": "fusion_test",
            "feature_cols": [],
        }
        vision = {"worker_detected": True, "distance_source": "vision", "worker_distance_est": 0.5, "confidence": 0.9}
        with patch("monitoring.services.control_center.predict_fusion", return_value=model) as predict:
            payload = assemble(vision, None)
        predict.assert_called_once()
        row = predict.call_args.args[0]
        self.assertEqual(row["worker_distance"], 0.5)
        self.assertEqual(payload["fusion"]["label"], "NORMAL")
        self.assertEqual(payload["fusion"]["confidence"], 0.62)
        self.assertEqual(payload["fusion"]["probabilities"]["NORMAL"], 0.62)
        self.assertEqual(payload["features"]["worker_distance"], 0.5)

    def test_a_person_without_a_distance_is_degraded(self):
        from unittest.mock import patch

        vision = {"worker_detected": True, "distance_source": "vision", "worker_distance_est": None}
        with patch("monitoring.services.control_center.predict_fusion") as predict:
            situation = classify(vision, None, machine_on=True)
        predict.assert_not_called()
        self.assertEqual(situation["label"], "DEGRADED")
        self.assertIn("distance", situation["reason"])


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
        self.assertEqual(
            log_line({"worker_detected": False, "distance_m": 3, "distance_source": "pinhole"}),
            "No person",
        )
        self.assertEqual(
            log_line({"worker_detected": True, "reason": "Possible interaction with machine."}),
            "Possible interaction with machine.",
        )

    def test_saved_boxes_keep_the_person_and_the_gear_corners(self):
        boxes = stored_boxes(
            {
                "detections": [
                    {
                        "xyxy": [10, 20, 110, 220],
                        "track_id": 2,
                        "is_primary": True,
                        "distance_m": 2.3777,
                        "distance_source": "pinhole",
                    },
                    {
                        "xyxy": [1, 1, 4, 4],
                        "track_id": 9,
                        "is_primary": False,
                        "distance_m": 4,
                        "distance_source": "heuristic",
                    },
                ],
                "ppe": {
                    "detections": [
                        {"xyxy": [12, 18, 40, 50], "track_id": 2, "label": "helmet", "missing": False},
                    ]
                },
            }
        )
        self.assertEqual(boxes[0]["kind"], "person")
        self.assertEqual(boxes[0]["track_id"], 2)
        self.assertEqual(boxes[0]["xyxy"], [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(boxes[0]["distance_m"], 2.378)
        self.assertIsNone(boxes[1]["distance_m"])
        self.assertEqual(boxes[2]["kind"], "gear")
        self.assertEqual(boxes[2]["label"], "helmet")
        self.assertEqual(boxes[2]["track_id"], 2)

    def test_the_log_on_screen_is_the_sample_at_that_time(self):
        offsets = [0.0, 2.0, 4.0]
        self.assertEqual(active_log_index(offsets, 0.0), 0)
        self.assertEqual(active_log_index(offsets, 3.9), 1)
        self.assertEqual(active_log_index(offsets, 4.0), 2)
        self.assertIsNone(active_log_index([], 1.0))

    def test_a_short_clip_is_logged_at_each_sample_second(self):
        from monitoring.services.recordings import _log_times

        times = _log_times(8.28)
        self.assertEqual(times, [0.0, 2.0, 4.0, 6.0, 8.0])


class FallAndSceneTests(SimpleTestCase):
    def _standing(self):
        return _pose({5: (100, 100, 1), 6: (140, 100, 1), 11: (105, 200, 1), 12: (135, 200, 1), 15: (105, 340, 1), 16: (135, 340, 1)})

    def _sitting_on_ground(self):
        return _pose({5: (160, 518, 1), 6: (165, 520, 1), 11: (210, 560, 1), 12: (216, 560, 1), 15: (280, 569, 1), 16: (286, 569, 1)})

    def test_a_standing_worker_is_not_down(self):
        from monitoring.services.accident_inference import body_lying

        self.assertFalse(body_lying(self._standing(), [90, 80, 150, 350]))

    def test_hips_at_ankle_height_is_down(self):
        from monitoring.services.accident_inference import body_lying

        self.assertTrue(body_lying(self._sitting_on_ground(), [110, 478, 302, 591]))

    def test_a_fall_box_on_a_standing_worker_is_not_a_fall(self):
        from monitoring.services.accident_inference import fall_for_box

        falls = {"available": True, "detections": [{"xyxy": [90, 80, 150, 350], "confidence": 0.9}]}
        self.assertFalse(fall_for_box([90, 80, 150, 350], falls, self._standing())["fallen"])
        self.assertTrue(fall_for_box([110, 478, 302, 591], {"available": True, "detections": [{"xyxy": [110, 478, 302, 591], "confidence": 0.9}]}, self._sitting_on_ground())["fallen"])

    def test_a_person_keeps_their_id_between_sparse_frames(self):
        from monitoring.services.person_tracker import IdKeeper

        keeper = IdKeeper()
        first = keeper.assign([[0, 0, 50, 100], [300, 0, 350, 100]], 0.0)
        moved = keeper.assign([[310, 5, 360, 105], [20, 0, 70, 100]], 0.25)
        self.assertEqual(moved, [first[1], first[0]])
        later = keeper.assign([[600, 0, 650, 100]], 3.0)
        self.assertNotIn(later[0], first)

    def test_scene_findings_raise_the_fusion_level(self):
        from monitoring.services.control_center import apply_scene_events, scene_events

        events = scene_events(fallen_ids=[3], smoke=True, behaviour={"unsafe": False, "label": "Safe Walkway"})
        level, reason = apply_scene_events("NORMAL", "", events)
        self.assertEqual(level, "DANGER")
        self.assertEqual(reason, "Fall: person 3 down · Smoke detected")
        unsafe = scene_events(behaviour={"unsafe": True, "label": "Opened Panel Cover"})
        self.assertEqual(apply_scene_events("DEGRADED", "Person seen, distance unavailable.", unsafe)[0], "CAUTION")
        self.assertEqual(apply_scene_events("DANGER", "", unsafe)[0], "DANGER")


class DistanceHonestyTests(SimpleTestCase):
    def _full_body(self):
        return _pose({0: (150, 40, 1), 15: (130, 340, 1), 16: (170, 340, 1)})

    def test_partial_body_has_no_range(self):
        self.assertIsNone(stature_px(_pose({0: (150, 40, 1)})))
        estimate = measure_person(_pose({0: (150, 40, 1)}), (320, 480), tof_m=1.2, sensor_aimed=False)
        self.assertIsNone(estimate)

    def test_full_body_uses_the_distance_network_until_the_sensor_is_on_that_person(self):
        body = self._full_body()
        estimate = measure_person(body, (320, 480), tof_m=1.2, sensor_aimed=False)
        self.assertEqual(estimate.source, "vision")
        self.assertTrue(str(estimate.method).startswith("residual_mlp:"))
        self.assertGreater(estimate.distance_m, 0)
        aimed = measure_person(body, (320, 480), tof_m=1.2, sensor_aimed=True)
        self.assertEqual(aimed.source, "tof")
        self.assertEqual(aimed.distance_m, 1.2)
        self.assertIsNotNone(aimed.body_m)

    def test_sensor_reading_is_not_given_to_someone_off_the_beam(self):
        self.assertFalse(aim_hits([0, 0, 10, 10], (100, 100)))
        self.assertTrue(aim_hits([40, 40, 60, 80], (100, 100)))


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


class BoardShakeTests(SimpleTestCase):
    def test_a_hard_knock_is_unusual_and_a_still_board_is_not(self):
        import numpy as np

        from monitoring.services.vibration_inference import predict_from_window, unusual_shake

        self.assertFalse(unusual_shake({"mag_std": 0.02}))
        self.assertTrue(unusual_shake({"mag_std": 0.6}))
        quiet = np.zeros((100, 3), dtype=np.float32)
        quiet[:, 2] = 1.0
        still = predict_from_window(quiet)
        self.assertEqual(still["decision"], "board")
        self.assertFalse(still["abnormal"])
        knock = quiet.copy()
        knock[40:48, 0] = 3.5
        knock[40:48, 1] = -2.0
        hit = predict_from_window(knock)
        self.assertTrue(hit["abnormal"])
        from unittest.mock import patch

        model = {
            "label": "CAUTION",
            "label_idx": 1,
            "confidence": 0.71,
            "probabilities": {"NORMAL": 0.1, "CAUTION": 0.71, "DANGER": 0.19},
            "model_id": "fusion_test",
            "feature_cols": [],
        }
        with patch("monitoring.services.control_center.predict_fusion", return_value=model) as predict:
            situation = classify({"worker_detected": False}, hit, machine_on=False)
        predict.assert_called_once()
        self.assertEqual(situation["label"], "CAUTION")
        self.assertEqual(predict.call_args.args[0]["vibration_state"], 1.0)


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

