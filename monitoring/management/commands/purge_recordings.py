"""Erase archived footage older than the retention period.

    python manage.py purge_recordings              # uses SAMS_RETENTION_DAYS (default 30)
    python manage.py purge_recordings --days 7 --dry-run

A recording survives only while one of its moments is marked "kept" for training
by a safety officer. Run it daily (cron or a Celery beat job) on the site server.
"""

from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from monitoring.models import Recording
from monitoring.services.recordings import delete_recording


class Command(BaseCommand):
    help = "Delete recordings older than the retention period, except those with moments kept for training."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=settings.SAMS_RETENTION_DAYS)
        parser.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting it.")

    def handle(self, *args, days: int, dry_run: bool, **options):
        cutoff = timezone.now() - timedelta(days=days)
        expired = (
            Recording.objects.filter(created_at__lt=cutoff)
            .exclude(render_status="pending")
            .exclude(minutes__train_status="kept")
            .distinct()
        )
        count = 0
        for recording in expired:
            count += 1
            if dry_run:
                self.stdout.write(f"would delete #{recording.pk} {recording.title} ({recording.created_at:%Y-%m-%d})")
            else:
                delete_recording(recording)
        verb = "Would delete" if dry_run else "Deleted"
        self.stdout.write(self.style.SUCCESS(f"{verb} {count} recording(s) older than {days} days."))
