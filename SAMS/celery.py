"""Celery app for SAMS background work (upload analysis, live video finalize)."""

from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SAMS.settings")

app = Celery("SAMS")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
