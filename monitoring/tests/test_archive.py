"""Archive, alerts, trials and the MLOps export, through the database and the review pages."""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from monitoring.services.control_center import decide

from .fixtures import COLS, FUSION_OUT, MEASURED_VISION, VIBRATION, VISION_FRAME, decided_payload, jpeg, live_payload, site_features


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
            decision = decide(MEASURED_VISION, VIBRATION, machine_on=None)
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


class TrialDataTests(TestCase):
    def setUp(self):
        from monitoring.services import trials

        self.trials = trials

    def test_without_a_trial_nothing_is_kept(self):
        from monitoring.models import TrainingSample

        self.assertIsNone(self.trials.record_live_samples(live_payload("NORMAL"), {}))
        self.assertEqual(TrainingSample.objects.count(), 0)

    def test_a_measured_poll_is_kept_under_the_trials_true_label(self):
        from monitoring.models import TrainingSample

        trial = self.trials.start_trial(risk_label="DANGER", scenario="walk in")
        vision = {"distance_pairs": [{"features": site_features(), "z0_m": 0.9, "distance_m": 0.62}]}
        status = self.trials.record_live_samples(live_payload("NORMAL"), vision)
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
        vision = {"distance_pairs": [{"features": site_features(), "z0_m": 0.9, "distance_m": 0.62}]}
        self.trials.record_live_samples(live_payload("CAUTION"), vision)
        with patch("monitoring.services.trials.get_feature_cols", return_value=COLS):
            payload = self.trials.export_zip()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            fusion = archive.read(self.trials.FUSION_CSV).decode().splitlines()
            site = archive.read(self.trials.SITE_DISTANCE_CSV).decode().splitlines()
        self.assertEqual(
            fusion[0].split(","),
            [*self.trials.FUSION_HEADER, *COLS, "label", "label_name", "predicted"],
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
        recorder.capture("tok", jpeg(10), decided_payload("NORMAL"), VISION_FRAME)
        recorder.capture("tok", jpeg(10), decided_payload("NORMAL"), VISION_FRAME)
        recorder.capture("tok", jpeg(200), decided_payload("DANGER"), VISION_FRAME)
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
        encode_clip([(0.0, jpeg(10)), (1.0, jpeg(200))], clip)
        recording = Recording.objects.create(title="clip.mp4", source="upload", declared={"machine_state": "RUNNING", "vibration": "NORMAL"})
        recording.video.save("clip.mp4", ContentFile(clip.read_bytes()), save=True)

        class Tracking:
            now = 0.0

        seen = []

        def fake_predict(image, *, annotate, video):
            seen.append(video.now)
            return dict(VISION_FRAME)

        with patch("monitoring.services.vision_inference.VideoTracking", Tracking), patch(
            "monitoring.services.vision_inference.predict_image", side_effect=fake_predict
        ), patch("monitoring.services.control_center.predict_fusion", return_value=FUSION_OUT) as fused, patch(
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
        rows = trials._fusion_csv(COLS).splitlines()
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
                {
                    "video": SimpleUploadedFile("yard.mp4", b"0" * 200, content_type="video/mp4"),
                    "machine_state": "STOPPED",
                    "next": "/archive/",
                },
            )
        recording = Recording.objects.get()
        self.assertRedirects(response, reverse("review_archive"), fetch_redirect_response=False)
        queued.assert_called_once_with(recording.pk)
        self.assertEqual(recording.declared, {"machine_state": "STOPPED"})
        self.assertEqual(recording.render_status, "pending")
