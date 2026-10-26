"""Celery wiring: upload and live finalize enqueue without a live Redis broker."""

from unittest.mock import patch

from django.test import SimpleTestCase, override_settings


@override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True)
class CeleryEnqueueTests(SimpleTestCase):
    def test_upload_enqueue_dispatches_the_analyze_task(self):
        from monitoring.services import video_analysis

        with patch("monitoring.tasks.analyze_upload_recording.delay") as delay:
            video_analysis.enqueue(42)
        delay.assert_called_once_with(42)

    def test_upload_enqueue_falls_back_to_a_thread_when_celery_fails(self):
        from monitoring.services import video_analysis

        with (
            patch("monitoring.tasks.analyze_upload_recording.delay", side_effect=ConnectionError("no broker")),
            patch("monitoring.services.video_analysis.threading.Thread") as thread,
        ):
            video_analysis.enqueue(7)
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()

    def test_live_finalize_dispatches_the_celery_task(self):
        from monitoring.services import live_recorder

        with patch("monitoring.tasks.finalize_live_recording.delay") as delay:
            live_recorder.enqueue_finalize(99)
        delay.assert_called_once_with(99)
