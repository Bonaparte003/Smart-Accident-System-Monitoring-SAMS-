"""Shared bits for the signed-in review pages."""

from __future__ import annotations

from monitoring.models import RecordingMinute

PROBLEM_LEVELS = ("CAUTION", "DANGER", "DEGRADED")


def review_nav(request):
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return {"alert_count": 0}
    count = RecordingMinute.objects.filter(level__in=PROBLEM_LEVELS).count()
    return {"alert_count": count}
