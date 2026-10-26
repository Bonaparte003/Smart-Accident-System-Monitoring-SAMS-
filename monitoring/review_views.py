"""Signed-in SAMS: home, archive, alerts, and the training desk."""

from __future__ import annotations

import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.db.models import Case, Count, IntegerField, Q, Value, When
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .models import RISK_LABELS, InteractionEvent, Recording, RecordingMinute, Trial
from .services import trials, video_analysis
from .services.mlops import export_zip, kept_minutes, model_cards
from .services.recordings import log_line
from .services.vision_inference import pose_overlay_config

PROBLEM_LEVELS = ("CAUTION", "DANGER", "DEGRADED")
PPE_ICONS = {
    "hard_hat": "fa-helmet-safety",
    "vest": "fa-vest",
    "gloves": "fa-mitten",
    "boots": "fa-shoe-prints",
    "goggles": "fa-glasses",
}
RANGES = {
    "24h": (timedelta(hours=24), "Number of detected events per hour"),
    "7d": (timedelta(days=7), "Number of detected events per day"),
    "30d": (timedelta(days=30), "Number of detected events per day"),
}


def _safe_next(raw: str) -> str:
    path = (raw or "").strip()
    if path.startswith("/") and not path.startswith("//"):
        return path
    return "/mlops/"


def _desk() -> dict:
    minutes = RecordingMinute.objects.all()
    return {
        "video_count": Recording.objects.count(),
        "minute_count": minutes.count(),
        "problem_count": minutes.filter(level__in=PROBLEM_LEVELS).count(),
        "kept_count": minutes.filter(train_status="kept").count(),
        "pending_count": minutes.filter(train_status="pending", level__in=PROBLEM_LEVELS).count(),
    }


def _event_time(mark: RecordingMinute):
    return mark.recording.created_at + timedelta(seconds=float(mark.offset_s or 0))


def _buckets(key: str, end):
    end = timezone.localtime(end)
    if key == "24h":
        hour = end.replace(minute=0, second=0, microsecond=0)
        starts = [hour - timedelta(hours=23 - index) for index in range(24)]
        step = timedelta(hours=1)
        show = {0, 4, 8, 12, 16, 20, 23}

        def label(moment):
            return moment.strftime("%H:%M")

    elif key == "7d":
        day = end.replace(hour=0, minute=0, second=0, microsecond=0)
        starts = [day - timedelta(days=6 - index) for index in range(7)]
        step = timedelta(days=1)
        show = set(range(7))

        def label(moment):
            return moment.strftime("%d %b")

    else:
        day = end.replace(hour=0, minute=0, second=0, microsecond=0)
        starts = [day - timedelta(days=29 - index) for index in range(30)]
        step = timedelta(days=1)
        show = set(range(0, 30, 5)) | {29}

        def label(moment):
            return moment.strftime("%d %b")

    return starts, step, show, label


def _chart(marks, key: str, end) -> dict:
    starts, step, _show, label = _buckets(key, end)
    series = {name: [0] * len(starts) for name in ("NORMAL", "CAUTION", "DANGER")}
    for mark in marks:
        moment = timezone.localtime(_event_time(mark))
        if moment < starts[0] or moment >= starts[-1] + step:
            continue
        index = min(len(starts) - 1, max(0, int((moment - starts[0]) / step)))
        bucket = series.get(mark.level)
        if bucket is not None:
            bucket[index] += 1
    return {
        "labels": [label(moment) for moment in starts],
        "normal": series["NORMAL"],
        "caution": series["CAUTION"],
        "danger": series["DANGER"],
    }


def _share(count: int, largest: int) -> int:
    if largest <= 0:
        return 0
    return max(8, round(100 * count / largest)) if count else 0


def _confidence(detail: dict) -> str:
    raw = (detail or {}).get("confidence")
    if raw is None:
        return "—"
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return "—"
    if number <= 1:
        number *= 100
    return f"{round(number)}%"


def _storage_detail() -> tuple[str, str]:
    """Video count and free space, both measured."""
    videos = Recording.objects.count()
    noun = "video" if videos == 1 else "videos"
    try:
        free = shutil.disk_usage(settings.MEDIA_ROOT).free / (1024**3)
        space = f"{free:.1f} GB free"
    except OSError:
        space = "space unavailable"
    return f"{videos} {noun}", space


def _health() -> list[dict]:
    """Live devices only. A sensor with no signal is not marked active."""
    from hardware import get_camera, get_stream
    from hardware.presence import get_presence

    board = get_presence().snapshot()
    link = get_stream().status()
    camera = get_camera().status()
    connected = bool(link.get("connected"))
    mpu_age = board.get("mpu_age_s")
    mpu_ok = connected and mpu_age is not None and mpu_age <= 2.0
    tof_age = board.get("tof_age_s")
    tof_ok = connected and board.get("tof_present") is True and tof_age is not None and tof_age <= 5.0
    buzzer_ok = connected and bool(board.get("buzzer_ack"))
    grab_age = camera.get("last_grab_age_s")
    camera_ok = bool(camera.get("has_frame")) and grab_age is not None and grab_age <= 8.0

    if camera_ok:
        camera_detail = "Phone sending frames"
    elif camera.get("phone_joined"):
        camera_detail = "Phone stalled"
    else:
        camera_detail = "No phone"
    if mpu_ok:
        accel_detail = link.get("serial_port") or "Samples arriving"
    elif link.get("last_error"):
        accel_detail = "Port did not open"
    else:
        accel_detail = "No serial data"
    if tof_ok and board.get("tof_m") is not None:
        distance_detail = f"{board['tof_m']:.2f} m"
    elif tof_ok:
        distance_detail = "Sensor on the board"
    else:
        distance_detail = "Not on the board"

    rows = [
        {
            "label": "Phone camera",
            "ok": camera_ok,
            "status": "Live" if camera_ok else "Not connected",
            "detail": camera_detail,
            "icon": "camera",
        },
        {
            "label": "Accelerometer",
            "ok": mpu_ok,
            "status": "Live" if mpu_ok else "Not connected",
            "detail": accel_detail,
            "icon": "accel",
        },
        {
            "label": "Distance sensor",
            "ok": tof_ok,
            "status": "Live" if tof_ok else "Not connected",
            "detail": distance_detail,
            "icon": "tof",
        },
        {
            "label": "Buzzer",
            "ok": buzzer_ok,
            "status": "Live" if buzzer_ok else "Not connected",
            "detail": "Board acknowledged" if buzzer_ok else "No acknowledgement",
            "icon": "buzzer",
        },
    ]
    stored, space = _storage_detail()
    rows.append(
        {
            "label": "Storage",
            "ok": space != "space unavailable",
            "status": stored,
            "detail": space,
            "icon": "storage",
        }
    )
    return rows


def _overview(raw_range: str) -> dict:
    key = raw_range if raw_range in RANGES else "24h"
    span, chart_caption = RANGES[key]
    end = timezone.now()
    start = end - span
    marks = list(
        RecordingMinute.objects.filter(recording__created_at__gte=start - span).select_related("recording")
    )

    def inside(mark, opening, closing) -> bool:
        moment = _event_time(mark)
        return opening <= moment < closing

    current = [mark for mark in marks if inside(mark, start, end)]

    def tally(rows, level=None) -> int:
        if level is None:
            return sum(1 for mark in rows if mark.level in PROBLEM_LEVELS)
        return sum(1 for mark in rows if mark.level == level)

    videos_total = Recording.objects.count()
    levels = Counter(mark.level for mark in current)
    normal = levels.get("NORMAL", 0)
    caution = levels.get("CAUTION", 0)
    danger = levels.get("DANGER", 0)
    total = normal + caution + danger

    def pct(count: int) -> int:
        return round(100 * count / total) if total else 0

    causes = Counter((mark.summary or "Other").strip() or "Other" for mark in current if mark.level in PROBLEM_LEVELS)
    ranked = causes.most_common(5)
    largest = ranked[0][1] if ranked else 0
    latest = (
        RecordingMinute.objects.filter(level__in=PROBLEM_LEVELS)
        .select_related("recording")
        .order_by("-recording__created_at", "-minute")[:5]
    )
    shots: dict[int, RecordingMinute] = {}
    recordings = list(
        Recording.objects.annotate(
            event_count=Count("minutes"),
            caution_count=Count("minutes", filter=Q(minutes__level="CAUTION")),
            danger_count=Count("minutes", filter=Q(minutes__level="DANGER")),
        ).order_by("-created_at")[:4]
    )
    if recordings:
        for mark in RecordingMinute.objects.filter(recording_id__in=[item.pk for item in recordings]).exclude(snapshot=""):
            shots.setdefault(mark.recording_id, mark)
    return {
        "overview": {
            "range": key,
            "caption": chart_caption,
            "videos": {"value": videos_total},
            "alerts": {"value": tally(current)},
            "caution": {"value": caution},
            "danger": {"value": danger},
            "chart": _chart(current, key, end),
            "distribution": {
                "total": total,
                "normal": normal,
                "caution": caution,
                "danger": danger,
                "normal_pct": pct(normal),
                "caution_pct": pct(caution),
                "danger_pct": pct(danger),
            },
            "causes": [{"label": label, "count": count, "share": _share(count, largest)} for label, count in ranked],
            "latest": [
                {
                    "time": timezone.localtime(_event_time(mark)).strftime("%H:%M"),
                    "source": mark.recording.title,
                    "detection": mark.summary or mark.level.title(),
                    "level": mark.level,
                    "confidence": _confidence(mark.detail),
                    "url_id": mark.recording_id,
                }
                for mark in latest
            ],
            "health": _health(),
            "recordings": [
                {
                    "id": item.pk,
                    "title": item.title,
                    "when": item.created_at,
                    "duration": item.duration_label,
                    "poster": shots[item.pk].snapshot.url if item.pk in shots else "",
                    "events": item.event_count,
                    "caution": item.caution_count,
                    "danger": item.danger_count,
                    "status": item.render_status,
                    "status_label": _status_label(item),
                    "processing": item.render_status == "pending",
                }
                for item in recordings
            ],
            "recordings_processing": any(item.render_status == "pending" for item in recordings),
        }
    }


@require_http_methods(["GET", "POST"])
def entrance(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return render(
            request,
            "monitoring/review/home.html",
            {"nav": "home", **_overview(request.GET.get("range") or "")},
        )

    error = ""
    username = ""
    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        password = request.POST.get("password") or ""
        user = authenticate(request, username=username, password=password)
        if user is None:
            error = "Those details were not recognized."
        else:
            login(request, user)
            return redirect("review_entrance")
    return render(
        request,
        "monitoring/review/login.html",
        {"error": error, "username": username},
    )


@require_http_methods(["POST"])
def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("review_entrance")


def _posters(recordings) -> dict[int, str]:
    ids = [item.pk for item in recordings]
    shots: dict[int, str] = {}
    if not ids:
        return shots
    for mark in RecordingMinute.objects.filter(recording_id__in=ids).exclude(snapshot="").order_by("minute"):
        shots.setdefault(mark.recording_id, mark.snapshot.url)
    return shots


def _created_bound(raw: str):
    """Parse a datetime-local value in the active timezone."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if timezone.is_naive(moment):
        moment = timezone.make_aware(moment, timezone.get_current_timezone())
    return moment


def _status_label(recording: Recording) -> str:
    if recording.render_status == "failed":
        return "analysis failed"
    if recording.render_status == "pending":
        return "recording" if recording.source == "live" else "analysing"
    return ""


@login_required(login_url="review_entrance")
def archive(request: HttpRequest) -> HttpResponse:
    created_from = (request.GET.get("from") or "").strip()
    created_to = (request.GET.get("to") or "").strip()
    severity = request.GET.get("severity") or "all"
    if severity not in {"all", "caution", "danger", "clear"}:
        severity = "all"
    recordings = Recording.objects.all()
    start = _created_bound(created_from)
    end = _created_bound(created_to)
    if start is not None:
        recordings = recordings.filter(created_at__gte=start)
    if end is not None:
        recordings = recordings.filter(created_at__lte=end)
    if severity == "caution":
        recordings = recordings.filter(minutes__level="CAUTION").distinct()
    elif severity == "danger":
        recordings = recordings.filter(minutes__level="DANGER").distinct()
    elif severity == "clear":
        recordings = recordings.filter(problem_count=0)
    recordings = list(recordings)
    posters = _posters(recordings)
    cards = [
        {
            "id": item.pk,
            "title": item.title,
            "poster": posters.get(item.pk, ""),
            "source": item.source,
            "status": item.render_status,
            "status_label": _status_label(item),
        }
        for item in recordings
    ]
    return render(
        request,
        "monitoring/review/archive.html",
        {
            "nav": "archive",
            "cards": cards,
            "cards_processing": any(item.render_status == "pending" for item in recordings),
            "created_from": created_from if start is not None else "",
            "created_to": created_to if end is not None else "",
            "severity": severity,
            **_desk(),
        },
    )


@login_required(login_url="review_entrance")
def alerts(request: HttpRequest) -> HttpResponse:
    rows = (
        RecordingMinute.objects.filter(level__in=PROBLEM_LEVELS)
        .select_related("recording")
        .order_by("-recording__created_at", "minute")
    )
    interactions = InteractionEvent.objects.all()[:80]
    return render(
        request,
        "monitoring/review/alerts.html",
        {"nav": "alerts", "rows": rows, "interactions": interactions, **_desk()},
    )


@login_required(login_url="review_entrance")
def mlops(request: HttpRequest) -> HttpResponse:
    # Every decided moment can become a training row; alerts come first.
    queue = (
        RecordingMinute.objects.filter(train_status="pending")
        .exclude(level="")
        .select_related("recording")
        .annotate(is_alert=Case(When(level__in=PROBLEM_LEVELS, then=Value(1)), default=Value(0), output_field=IntegerField()))
        .order_by("-is_alert", "-recording__created_at", "minute")[:24]
    )
    return render(
        request,
        "monitoring/review/mlops.html",
        {
            "nav": "mlops",
            "models": model_cards(),
            "queue": queue,
            "active_trial": trials.active_trial(),
            "trials": trials.with_counts(Trial.objects.all())[:12],
            "risk_labels": RISK_LABELS,
            **_desk(),
        },
    )


@login_required(login_url="review_entrance")
@require_POST
def start_trial(request: HttpRequest) -> HttpResponse:
    try:
        trials.start_trial(
            risk_label=request.POST.get("risk_label") or "",
            scenario=request.POST.get("scenario") or "",
            participant=request.POST.get("participant") or "",
            notes=request.POST.get("notes") or "",
        )
    except ValueError:
        pass
    return redirect(_safe_next(request.POST.get("next") or ""))


@login_required(login_url="review_entrance")
@require_POST
def stop_trial(request: HttpRequest) -> HttpResponse:
    trials.end_trial()
    return redirect(_safe_next(request.POST.get("next") or ""))


@login_required(login_url="review_entrance")
@require_POST
def delete_trial(request: HttpRequest, pk: int) -> HttpResponse:
    trials.delete_trial(get_object_or_404(Trial, pk=pk))
    return redirect(_safe_next(request.POST.get("next") or ""))


@login_required(login_url="review_entrance")
def export_trials(request: HttpRequest) -> HttpResponse:
    response = HttpResponse(trials.export_zip(), content_type="application/zip")
    response["Content-Disposition"] = 'attachment; filename="sams-trial-data.zip"'
    return response


@login_required(login_url="review_entrance")
@require_POST
def label_minute(request: HttpRequest) -> HttpResponse:
    mark = get_object_or_404(RecordingMinute, pk=request.POST.get("minute_id"))
    status = request.POST.get("status") or ""
    if status in {"pending", "kept", "excluded"}:
        mark.train_status = status
        true_label = request.POST.get("true_label") or ""
        if true_label in dict(RISK_LABELS):
            mark.true_label = true_label
        mark.save(update_fields=["train_status", "true_label"])
    return redirect(_safe_next(request.POST.get("next") or ""))


@login_required(login_url="review_entrance")
def export_training(request: HttpRequest) -> HttpResponse:
    scope = request.GET.get("scope") or "kept"
    if scope == "all":
        minutes = RecordingMinute.objects.select_related("recording").order_by("recording_id", "minute")
        filename = "sams-archive-labels.zip"
    else:
        minutes = kept_minutes()
        filename = "sams-training-set.zip"
    payload = export_zip(minutes)
    response = HttpResponse(payload, content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@login_required(login_url="review_entrance")
@require_POST
def upload_video(request: HttpRequest) -> HttpResponse:
    """Keep an uploaded clip and analyse it in the background with what the operator declared."""
    upload = request.FILES.get("video")
    if not upload:
        return redirect("review_archive")
    title = Path(upload.name or "clip.mp4").name.replace("\x00", "") or "clip.mp4"
    recording = Recording.objects.create(
        title=title[:200],
        source="upload",
        declared=video_analysis.clean_declared(request.POST.get("machine_state"), request.POST.get("vibration")),
        render_status="pending",
    )
    recording.video.save(title, upload, save=True)
    video_analysis.enqueue(recording.pk)
    # Stay on the list so the new clip shows as processing — do not open the detail page.
    next_path = (request.POST.get("next") or "").strip()
    if next_path.startswith("/") and not next_path.startswith("//"):
        if next_path.rstrip("/") in {"", "/"} or next_path.startswith("/?"):
            return redirect("review_entrance")
        if next_path.startswith("/archive"):
            return redirect("review_archive")
    return redirect("review_archive")


@login_required(login_url="review_entrance")
def live(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "monitoring/review/live.html",
        {"nav": "home", "ppe_icons": PPE_ICONS, **_desk()},
    )


@login_required(login_url="review_entrance")
def recording_status(request: HttpRequest, pk: int) -> JsonResponse:
    recording = get_object_or_404(Recording, pk=pk)
    return JsonResponse(
        {
            "status": recording.render_status,
            "framed": recording.framed.url if recording.framed else "",
            "source": recording.video.url if recording.video else "",
        }
    )


@login_required(login_url="review_entrance")
def recording_detail(request: HttpRequest, pk: int) -> HttpResponse:
    recording = get_object_or_404(Recording.objects.prefetch_related("minutes"), pk=pk)
    duration = float(recording.duration_s or 0)
    minutes = []
    for mark in recording.minutes.all():
        detail = mark.detail or {}
        if duration:
            pct = min(100.0, max(0.0, float(mark.offset_s or 0) / duration * 100))
        else:
            pct = 0.0
        raw_confidence = detail.get("confidence")
        try:
            confidence_pct = None if raw_confidence is None else round(float(raw_confidence) * 100)
        except (TypeError, ValueError):
            confidence_pct = None
        minutes.append(
            {
                "id": mark.pk,
                "minute": mark.minute,
                "offset_s": mark.offset_s,
                "pct": round(pct, 2),
                "clock": mark.clock,
                "level": mark.level,
                "problem": mark.problem,
                "summary": mark.summary,
                "line": log_line(detail, mark.level) or mark.summary,
                "reason": detail.get("reason") or "",
                "people": [
                    person
                    for person in (detail.get("people") or [])
                    if isinstance(person, dict)
                ],
                "snapshot": mark.snapshot.url if mark.snapshot else "",
                "vision": detail.get("vision") or "—",
                "confidence": raw_confidence,
                "confidence_pct": confidence_pct,
                "worker_detected": detail.get("worker_detected"),
                "distance_m": detail.get("distance_m"),
                "distance_source": detail.get("distance_source") or "",
                "person_count": detail.get("person_count"),
                "frame_w": detail.get("frame_w") or 0,
                "frame_h": detail.get("frame_h") or 0,
                "boxes": [
                    box
                    for box in (detail.get("boxes") or [])
                    if isinstance(box, dict) and isinstance(box.get("xyxy"), list)
                ],
                "ppe": [
                    {
                        **item,
                        "icon": PPE_ICONS.get(item.get("key"), "fa-user"),
                    }
                    for item in (detail.get("ppe") or [])
                    if isinstance(item, dict)
                ],
                "train_status": mark.train_status,
            }
        )
    play_url = recording.video.url if recording.video else ""
    baked = recording.render_status == "ready" and bool(recording.framed)
    if baked:
        play_url = recording.framed.url
    return render(
        request,
        "monitoring/review/detail.html",
        {
            "nav": "archive",
            "recording": recording,
            "minutes": minutes,
            "play_url": play_url,
            "baked": baked,
            "status_label": _status_label(recording),
            "pose_overlay": pose_overlay_config(),
        },
    )
