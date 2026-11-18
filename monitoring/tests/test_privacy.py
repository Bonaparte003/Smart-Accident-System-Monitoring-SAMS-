"""Privacy safeguards: public notice, erasure of a recording, and the retention purge."""

import shutil
import tempfile
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from monitoring.models import Recording, RecordingMinute

from .fixtures import jpeg


class PrivacyTests(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        override = override_settings(MEDIA_ROOT=self.media)
        override.enable()
        self.addCleanup(override.disable)
        self.user = User.objects.create_user("officer", password="sams-review")

    def _recording(self, *, days_old=0, kept=False):
        recording = Recording.objects.create(title="yard.mp4", render_status="ready", duration_s=4)
        recording.video.save("yard.mp4", ContentFile(b"video"), save=True)
        minute = RecordingMinute.objects.create(
            recording=recording, minute=0, level="CAUTION", train_status="kept" if kept else "pending"
        )
        minute.snapshot.save("f.jpg", ContentFile(jpeg(120)), save=True)
        if days_old:
            Recording.objects.filter(pk=recording.pk).update(created_at=timezone.now() - timedelta(days=days_old))
        return recording, minute

    def test_privacy_page_is_public(self):
        response = self.client.get(reverse("review_privacy"))
        self.assertContains(response, "Privacy notice and terms of use")
        self.assertContains(response, "deleted automatically after")

    def test_delete_needs_sign_in(self):
        recording, _ = self._recording()
        self.client.post(reverse("review_recording_delete", args=[recording.pk]))
        self.assertTrue(Recording.objects.filter(pk=recording.pk).exists())

    def test_delete_erases_rows_and_files(self):
        recording, minute = self._recording()
        video_path, frame_path = recording.video.path, minute.snapshot.path
        self.client.login(username="officer", password="sams-review")
        response = self.client.post(reverse("review_recording_delete", args=[recording.pk]))
        self.assertRedirects(response, reverse("review_archive"))
        self.assertFalse(Recording.objects.filter(pk=recording.pk).exists())
        self.assertFalse(RecordingMinute.objects.filter(pk=minute.pk).exists())
        import os

        self.assertFalse(os.path.exists(video_path))
        self.assertFalse(os.path.exists(frame_path))

    def test_purge_removes_old_footage_but_keeps_reviewed_training_moments(self):
        old, _ = self._recording(days_old=40)
        kept, _ = self._recording(days_old=40, kept=True)
        fresh, _ = self._recording(days_old=2)
        call_command("purge_recordings", days=30, stdout=open("/dev/null", "w"))
        remaining = set(Recording.objects.values_list("pk", flat=True))
        self.assertEqual(remaining, {kept.pk, fresh.pk})
        self.assertNotIn(old.pk, remaining)
