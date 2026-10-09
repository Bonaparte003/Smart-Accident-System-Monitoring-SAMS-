import json
import logging
import os

from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from hardware import get_buzzer, get_camera, get_leds, get_stream, get_tof
from hardware.stream import apply_risk_actuators
from hardware.wifi_camera import guess_lan_ip


from .services import event_log
from .services.interaction_log import record_interaction
from .services.recordings import log_line_from_prediction
from .services.control_center import assemble as assemble_control
from .services.control_center import decide_alert
from .services.feature_builder import build_features, build_simulation_response
from .services.fusion_inference import (
    CAUTION_ZONE_M,
    CLASS_NAMES,
    DANGER_ZONE_M,
    FEATURE_COLS,
    get_metadata,
    get_model,
    predict_fusion,
)
from .services import distance_estimation
from .services import ppe_inference as ppe
from .services import vibration_inference as vib
from .services import vision_inference as vision

logger = logging.getLogger("monitoring")


def _live_vibration_result():
    """Run vibration model on the latest MPU6050 window, or None if not ready."""
    stream = get_stream()
    win = stream.latest_window()
    if win is None:
        return None
    result = vib.predict_from_window(win, controls=None)
    result["source"] = "live"
    result["hardware"] = stream.status()
    stream.record_prediction(result)
    return result


def app(request):
    """Main SAMS shell. Control Center is the joined safety decision."""
    return render(request, "monitoring/app.html")


@require_http_methods(["GET", "HEAD"])
def favicon(request):
    return HttpResponseRedirect(static("monitoring/favicon.svg") + "?v=plus")


@require_GET
def health(request):
    try:
        get_model()
        vib.get_model()
        meta = get_metadata()
        return JsonResponse(
            {
                "status": "ok",
                "fusion_model": meta.get("experiment_id"),
                "vibration_model": vib.get_metadata().get("experiment_id"),
                "vision": vision.get_torch_status(),
                "classes": CLASS_NAMES,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("health check failed")
        return JsonResponse({"status": "error", "detail": str(exc)}, status=500)


@require_GET
def config(request):
    meta = get_metadata()
    return JsonResponse(
        {
            "class_names": CLASS_NAMES,
            "feature_cols": FEATURE_COLS,
            "zones_m": {"caution": CAUTION_ZONE_M, "danger": DANGER_ZONE_M},
            "fusion_model": {
                "experiment_id": meta.get("experiment_id"),
                "model": meta.get("model"),
                "approach": meta.get("approach"),
                "test_metrics": {
                    k: meta.get("test_metrics", {}).get(k)
                    for k in ("accuracy", "f1_macro", "danger_recall", "false_alarm_rate")
                },
            },
            "vibration_model": {
                "experiment_id": vib.get_metadata().get("experiment_id"),
                "class_names": vib.CLASS_NAMES,
                "window_size": vib.WINDOW_SIZE,
                "sample_rate_hz": vib.SAMPLE_RATE_HZ,
            },
            "vision_model": vision.get_torch_status(),
            "scenarios": [
                {"id": "normal_pass", "label": "Normal pass"},
                {"id": "cautious_approach", "label": "Cautious approach"},
                {"id": "slow_approach", "label": "Slow approach"},
                {"id": "fast_approach", "label": "Fast approach"},
                {"id": "danger_entry", "label": "Danger-zone entry"},
            ],
            "defaults": {
                "worker_distance": 1.8,
                "approaching_machine": True,
                "machine_active": True,
                "abnormal_vibration": False,
            },
            "hardware": get_stream().status(),
            "buzzer": get_buzzer().status(),
            "leds": get_leds().status(),
            "camera": get_camera().status(),
            "logging": {
                "ui": "Activity Log panel in the app (left nav → Logs)",
                "server": "Django console where `runserver` is running",
                "api": "GET /api/logs/",
            },
        }
    )


def _parse_body(request) -> dict:
    if request.content_type and "application/json" in request.content_type:
        try:
            return json.loads(request.body.decode() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    return {k: request.POST.get(k) for k in request.POST.keys()}


def _coerce_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@csrf_exempt
@require_http_methods(["POST"])
def simulate(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        distance = float(body.get("worker_distance", 1.8))
    except (TypeError, ValueError):
        return JsonResponse({"error": "worker_distance must be a number"}, status=400)

    stream = get_stream()
    camera = get_camera()
    live_vib = None
    if stream.is_live:
        live_vib = _live_vibration_result()
        abnormal = bool(live_vib and live_vib.get("abnormal"))
        vib_score = float(live_vib["anomaly_score"]) if live_vib else 0.12
    else:
        abnormal = _coerce_bool(body.get("abnormal_vibration"), False)
        vib_score = None

    live_vis = camera.last_prediction() if camera.is_live else None
    worker_detected = (
        _coerce_bool(body["worker_detected"], True)
        if "worker_detected" in body
        else (bool(live_vis.get("worker_detected")) if live_vis else None)
    )
    vision_confidence = (
        float(body["vision_confidence"])
        if body.get("vision_confidence") is not None
        else (float(live_vis["confidence"]) if live_vis else None)
    )
    if (
        "worker_distance" not in body
        and live_vis
        and live_vis.get("worker_detected")
        and live_vis.get("worker_distance_est") is not None
    ):
        distance = float(live_vis["worker_distance_est"])

    live_ppe = (live_vis or {}).get("ppe") if live_vis else None
    hard_hat = _coerce_bool(body.get("hard_hat"), True)
    vest = _coerce_bool(body.get("vest"), True)
    gloves = _coerce_bool(body.get("gloves"), True)
    boots = _coerce_bool(body.get("boots"), True)
    if live_ppe and live_ppe.get("available") and live_ppe.get("source") == "vision_ppe":
        if "hard_hat" not in body and live_ppe.get("hard_hat") is not None:
            hard_hat = bool(live_ppe["hard_hat"])
        if "vest" not in body and live_ppe.get("vest") is not None:
            vest = bool(live_ppe["vest"])
        if "gloves" not in body and live_ppe.get("gloves") is not None:
            gloves = bool(live_ppe["gloves"])
        if "boots" not in body and live_ppe.get("boots") is not None:
            boots = bool(live_ppe["boots"])

    features = build_features(
        worker_distance=distance,
        approaching_machine=_coerce_bool(body.get("approaching_machine"), True),
        machine_active=_coerce_bool(body.get("machine_active"), True),
        abnormal_vibration=abnormal,
        hard_hat=hard_hat,
        vest=vest,
        gloves=gloves,
        boots=boots,
        worker_detected=worker_detected,
        worker_speed=float(body["worker_speed"]) if body.get("worker_speed") is not None else None,
        vision_confidence=vision_confidence,
    )
    if vib_score is not None:
        features["vibration_anomaly_score"] = vib_score
        features["vibration_state"] = 1.0 if abnormal else 0.0

    fusion = predict_fusion(features)
    if stream.is_live:
        apply_risk_actuators(
            fusion["label"],
            buzzer=stream.buzzer_auto,
            leds=stream.leds_auto,
        )

    payload = build_simulation_response(features, fusion)
    payload["hardware"] = stream.status()
    payload["camera"] = camera.status()
    if live_vib:
        payload["live_vibration"] = {
            "label": live_vib["label"],
            "confidence": live_vib["confidence"],
            "abnormal": live_vib["abnormal"],
            "anomaly_score": live_vib["anomaly_score"],
        }
    if live_vis:
        payload["live_vision"] = {
            "label": live_vis["label"],
            "confidence": live_vis["confidence"],
            "unsafe": live_vis.get("unsafe", False),
            "worker_detected": bool(live_vis.get("worker_detected")),
            "person_count": int(live_vis.get("person_count") or 0),
            "worker_distance_est": live_vis.get("worker_distance_est"),
            "distance_source": live_vis.get("distance_source"),
            "distance_method": live_vis.get("distance_method"),
            "tof_m": live_vis.get("tof_m"),
            "ppe": live_vis.get("ppe"),
            "ppe_summary": live_vis.get("ppe_summary"),
            "ppe_complete": live_vis.get("ppe_complete"),
        }
        payload["hud"]["vision"]["status"] = live_vis["label"]
        payload["hud"]["vision"]["confidence"] = round(float(live_vis["confidence"]), 3)
        if live_vis.get("ppe") and live_vis["ppe"].get("available"):
            live_ppe = live_vis["ppe"]
            payload["hud"]["ppe"] = {
                "summary": live_ppe.get("summary"),
                "complete": live_ppe.get("ppe_complete"),
                "missing": live_ppe.get("missing") or [],
                "items": {
                    k: {
                        "label": (live_ppe.get("items") or {}).get(k, {}).get("label", k),
                        "worn": live_ppe.get(k),
                        "status": (live_ppe.get("items") or {}).get(k, {}).get("status", "—"),
                    }
                    for k in ("hard_hat", "vest", "gloves", "boots")
                },
                "source": "vision_ppe",
            }
            payload["hud"]["hard_hat"] = (live_ppe.get("items") or {}).get("hard_hat", {}).get(
                "status", "—"
            )
            if live_ppe.get("summary"):
                payload["hud"]["vision"]["status"] = (
                    f"{live_vis['label']} · {live_ppe['summary']}"
                )
    tags = []
    if stream.is_live:
        tags.append("live vib")
    if live_vis:
        tags.append("live cam")
    event_log.log_event(
        "info",
        "fusion",
        f"Simulate → {fusion['label']} ({fusion['confidence']:.0%})"
        + (f" [{', '.join(tags)}]" if tags else ""),
        distance=distance,
        label=fusion["label"],
        confidence=fusion["confidence"],
    )
    logger.info("simulate label=%s conf=%.3f dist=%.2f", fusion["label"], fusion["confidence"], distance)
    return JsonResponse(payload)


@csrf_exempt
@require_http_methods(["POST"])
def predict_fusion_api(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    missing = [c for c in FEATURE_COLS if c not in body]
    if missing:
        return JsonResponse({"error": "Missing feature columns", "missing": missing}, status=400)

    try:
        features = {c: float(body[c]) for c in FEATURE_COLS}
    except (TypeError, ValueError):
        return JsonResponse({"error": "All features must be numeric"}, status=400)

    fusion = predict_fusion(features)
    event_log.log_event("info", "fusion", f"Predict → {fusion['label']}", **fusion["probabilities"])
    return JsonResponse(build_simulation_response(features, fusion))


def _control_payload(vision: dict | None, *, actuate: bool) -> dict:
    stream = get_stream()
    vibration = _live_vibration_result() if stream.is_live else None
    payload = assemble_control(vision, vibration, actuate=actuate and stream.is_live)
    payload["hardware"] = stream.status()
    payload["tof"] = _tof_status()
    if vibration is not None:
        payload["vibration"]["series"] = vibration.get("series")
    else:
        payload["vibration"] = {
            "label": None,
            "ready": False,
            "detail": "Accelerometer not streaming",
        }
    return payload


@csrf_exempt
@require_http_methods(["POST"])
def control_fuse(request):
    """Fusion for one already-classified frame (upload path)."""
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    vision = body.get("vision") if isinstance(body.get("vision"), dict) else body
    payload = _control_payload(vision, actuate=False)
    event_log.log_event(
        "info",
        "control",
        f"{payload['fusion']['label']} · {log_line_from_prediction(vision, payload['fusion']['label'])}",
        label=payload["fusion"]["label"],
    )
    return JsonResponse(payload)


@require_GET
def control_live(request):
    """Poll: classify the phone frame, read vibration, run fusion."""
    camera = get_camera()
    if not camera.is_live:
        return JsonResponse(
            {
                "ready": False,
                "error": "Scan the QR to start SAMS live.",
                "camera": camera.status(),
            }
        )
    try:
        vision = _live_vision_result(classify=True)
    except Exception as exc:
        return JsonResponse(
            {
                "ready": False,
                "error": str(exc),
                "camera": camera.status(),
            }
        )
    payload = _control_payload(vision, actuate=True)
    record_interaction(payload)
    decision = (payload.get("fusion") or {}).get("label") or "NORMAL"
    reason = (payload.get("fusion") or {}).get("reason") or ""
    line = reason or log_line_from_prediction(vision, decision)
    message = f"{decision} · {line}" if line else decision
    if getattr(control_live, "_last", None) != message:
        control_live._last = message
        level = "error" if decision == "DANGER" else "warning" if decision in {"CAUTION", "DEGRADED"} else "info"
        event_log.log_event(level, "control", message, label=decision, reason=reason)
    payload["ready"] = True
    payload["camera"] = camera.status()
    return JsonResponse(payload)


@csrf_exempt
@require_http_methods(["POST"])
def vibration_predict(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    stream = get_stream()
    force_live = _coerce_bool(body.get("live"), stream.is_live)

    if force_live:
        result = _live_vibration_result()
        if result is None:
            return JsonResponse(
                {
                    "error": "Live window not ready yet",
                    "hardware": stream.status(),
                },
                status=503,
            )
        event_log.log_event(
            "warning" if result["abnormal"] else "info",
            "vibration",
            f"Live MPU6050 → {result['label']} ({result['confidence']:.0%})",
            label=result["label"],
            hw_source="live",
        )
        logger.info("vibration live label=%s", result["label"])
        return JsonResponse(result)

    try:
        x_amp = float(body.get("x_amp", 0.05))
        y_amp = float(body.get("y_amp", 0.05))
        z_amp = float(body.get("z_amp", 0.05))
        freq_hz = float(body.get("freq_hz", 5.0))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Amplitudes must be numeric"}, status=400)

    result = vib.predict_from_amplitudes(x_amp=x_amp, y_amp=y_amp, z_amp=z_amp, freq_hz=freq_hz)
    result["source"] = "simulate"
    event_log.log_event(
        "warning" if result["abnormal"] else "info",
        "vibration",
        f"Vibration → {result['label']} ({result['confidence']:.0%})",
        x_amp=x_amp,
        y_amp=y_amp,
        z_amp=z_amp,
        label=result["label"],
    )
    logger.info("vibration label=%s x=%.3f y=%.3f", result["label"], x_amp, y_amp)
    return JsonResponse(result)


@require_GET
def vibration_live(request):
    """Poll the latest live MPU6050 window prediction, plus the ToF reading."""
    stream = get_stream()
    tof = _tof_status()
    if not stream.is_live:
        return JsonResponse({"error": "Not in live mode", "hardware": stream.status(), "tof": tof}, status=400)
    cached = stream.last_prediction()
    result = _live_vibration_result()
    if result is None:
        return JsonResponse(
            {
                "ready": False,
                "hardware": stream.status(),
                "last": cached,
                "tof": tof,
            }
        )
    return JsonResponse({"ready": True, **result, "tof": tof})


def _tof_status() -> dict:
    status = get_tof().status()
    try:
        status["near_m"] = float(distance_estimation.load_calib()["tof_near_m"])
    except (OSError, ValueError, KeyError):
        status["near_m"] = None
    return status


@require_GET
def hardware_status(request):
    return JsonResponse(
        {
            "hardware": get_stream().status(),
            "buzzer": get_buzzer().status(),
            "leds": get_leds().status(),
            "camera": get_camera().status(),
        }
    )


def _public_join_base(request) -> str:
    """URL the phone can open (LAN IP + HTTPS when available)."""
    override = os.environ.get("SAMS_PUBLIC_BASE", "").strip().rstrip("/")
    if override:
        return override
    want_https = (
        request.is_secure()
        or os.environ.get("SAMS_HTTPS", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    scheme = "https" if want_https else request.scheme
    ip = guess_lan_ip()
    port = request.get_port()
    return f"{scheme}://{ip}:{port}"


def _apply_live_ppe(pred: dict) -> None:
    """Copy the per-person PPE decision onto the flat payload fields."""
    stable = pred.get("ppe")
    if not isinstance(stable, dict):
        return
    pred["hard_hat"] = stable.get("hard_hat")
    pred["vest"] = stable.get("vest")
    pred["gloves"] = stable.get("gloves")
    pred["boots"] = stable.get("boots")
    pred["ppe_complete"] = stable.get("ppe_complete")
    pred["ppe_summary"] = stable.get("summary")
    pred["unsafe"] = stable.get("ppe_complete") is False


def _live_vision_result(*, classify: bool = True) -> dict:
    """Use latest phone-pushed frame and optionally run YOLOv8 person detection."""
    camera = get_camera()
    image = camera.grab_image()
    thumb = camera.last_thumbnail_data_url()
    result: dict = {
        "source": "live",
        "camera": camera.status(),
        "thumbnail": thumb,
    }
    if not classify:
        return result
    pred = vision.predict_image(image, live=True)
    _apply_live_ppe(pred)
    camera.record_prediction(
        {
            "label": pred["label"],
            "confidence": pred["confidence"],
            "worker_detected": pred["worker_detected"],
            "person_count": pred["person_count"],
            "worker_distance_est": pred.get("worker_distance_est"),
            "distance_source": pred.get("distance_source"),
            "distance_method": pred.get("distance_method"),
            "distance_confidence": pred.get("distance_confidence"),
            "z0_m": pred.get("z0_m"),
            "tof_m": pred.get("tof_m"),
            "ppe": {
                "available": (pred.get("ppe") or {}).get("available"),
                "source": (pred.get("ppe") or {}).get("source"),
                "summary": pred.get("ppe_summary"),
                "ppe_complete": pred.get("ppe_complete"),
                "missing": (pred.get("ppe") or {}).get("missing"),
                "hard_hat": pred.get("hard_hat"),
                "vest": pred.get("vest"),
                "gloves": pred.get("gloves"),
                "boots": pred.get("boots"),
                "items": (pred.get("ppe") or {}).get("items"),
            },
            "ppe_summary": pred.get("ppe_summary"),
            "ppe_complete": pred.get("ppe_complete"),
            "unsafe": pred.get("unsafe", False),
            "model_id": pred.get("model_id"),
            "track_id": pred.get("track_id"),
            "worker_speed": pred.get("worker_speed"),
            "approach_velocity": pred.get("approach_velocity"),
            "approaching": pred.get("approaching"),
            "fall_detected": pred.get("fall_detected"),
            "fire_score": pred.get("fire_score"),
            "smoke_score": pred.get("smoke_score"),
        }
    )
    result.update(pred)
    if pred.get("annotated_thumbnail"):
        result["thumbnail"] = pred["annotated_thumbnail"]
    result["camera"] = camera.status()
    return result


@csrf_exempt
@require_http_methods(["POST", "GET"])
def hardware_camera(request):
    """Start/stop QR join session for the phone camera."""
    camera = get_camera()
    if request.method == "GET":
        return JsonResponse({"camera": camera.status(), "lan_ip": guess_lan_ip()})

    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    action = str(body.get("action") or "").strip().lower()
    if action == "start" or _coerce_bool(body.get("live"), False) or _coerce_bool(body.get("start"), False):
        ppe.reset_live_votes()
        vision.reset_tracks()
        base = _public_join_base(request)
        status = camera.start_session(join_base=base)
        event_log.log_event("info", "vision", "Camera QR session started — waiting for phone")
        return JsonResponse({"ok": True, "camera": status, "join_url": status.get("join_url")})

    if action == "stop" or ("live" in body and not _coerce_bool(body.get("live"), True)):
        ppe.reset_live_votes()
        vision.reset_tracks()
        status = camera.stop_session()
        event_log.log_event("info", "vision", "Camera QR session stopped")
        return JsonResponse({"ok": True, "camera": status})

    return JsonResponse({"camera": camera.status()})


@require_GET
def vision_live(request):
    """Poll: detect people in the latest frame pushed by the phone."""
    camera = get_camera()
    if not camera.is_live:
        return JsonResponse(
            {"ready": False, "error": "Camera session not active — show the QR again", "camera": camera.status()},
            status=200,
        )
    prev = camera.last_prediction()
    try:
        result = _live_vision_result(classify=True)
    except Exception as exc:
        return JsonResponse(
            {
                "ready": False,
                "error": str(exc),
                "camera": camera.status(),
                "last": prev,
                "torch": vision.get_torch_status(),
            },
            status=200,
        )

    stream = get_stream()
    if stream.is_live and (stream.buzzer_auto or stream.leds_auto):
        apply_risk_actuators(
            decide_alert(result, _live_vibration_result()),
            buzzer=stream.buzzer_auto,
            leds=stream.leds_auto,
        )

    line = log_line_from_prediction(result)
    if not prev or log_line_from_prediction(prev) != line:
        event_log.log_event(
            "warning" if result.get("unsafe") else "info",
            "vision",
            f"Vision · {line}",
            label=result["label"],
            unsafe=result.get("unsafe"),
        )
    return JsonResponse({"ready": True, **result})


def phone_cam_join(request, token: str):
    """Mobile page opened when the phone scans the QR."""
    camera = get_camera()
    ok = camera.validate_token(token)
    return render(
        request,
        "monitoring/phone_cam.html",
        {
            "token": token,
            "valid": ok,
            "join_url": camera.join_url(),
        },
    )


@csrf_exempt
@require_http_methods(["POST"])
def phone_cam_join_ack(request, token: str):
    camera = get_camera()
    try:
        camera.mark_joined(token)
    except PermissionError as exc:
        return JsonResponse({"error": str(exc)}, status=403)
    return JsonResponse({"ok": True, "camera": camera.status()})


@csrf_exempt
@require_http_methods(["POST"])
def phone_cam_frame(request, token: str):
    """Phone uploads a JPEG frame (multipart `frame` or raw body)."""
    camera = get_camera()
    data = None
    if request.FILES.get("frame"):
        data = request.FILES["frame"].read()
    elif request.body:
        data = request.body
    if not data:
        return JsonResponse({"error": "Send JPEG as multipart field `frame`"}, status=400)
    try:
        status = camera.push_jpeg(token, data)
    except PermissionError as exc:
        return JsonResponse({"error": str(exc)}, status=403)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"ok": True, "camera": status})


@csrf_exempt
@require_http_methods(["POST"])
def hardware_mode(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    mode = str(body.get("mode", "")).strip().lower()
    stream = get_stream()
    if "machine_on" in body and not mode:
        stream.set_machine_on(_coerce_bool(body.get("machine_on"), False))
        event_log.log_event(
            "info",
            "hardware",
            "Machine on" if stream.machine_on else "Machine off",
            machine_on=stream.machine_on,
        )
        return JsonResponse(
            {
                "hardware": stream.status(),
                "buzzer": get_buzzer().status(),
                "leds": get_leds().status(),
            }
        )
    if not mode:
        mode = "simulate"
    try:
        status = stream.set_mode(mode)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"error": str(exc), "hardware": get_stream().status()}, status=503)

    event_log.log_event(
        "info",
        "hardware",
        f"Mode → {status['mode']}",
        mode=status.get("mode"),
        connected=status.get("connected"),
        serial_port=status.get("serial_port"),
    )
    return JsonResponse(
        {
            "hardware": status,
            "buzzer": get_buzzer().status(),
            "leds": get_leds().status(),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def hardware_buzzer(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    buzzer = get_buzzer()
    stream = get_stream()
    if "enabled" in body:
        buzzer.set_enabled(_coerce_bool(body.get("enabled"), True))
    if "auto" in body:
        stream.set_buzzer_auto(_coerce_bool(body.get("auto"), True))
    if "state" in body:
        try:
            buzzer.set_state(str(body["state"]).strip().lower())
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
    if "label" in body:
        buzzer.apply_label(str(body["label"]))

    return JsonResponse({"buzzer": buzzer.status(), "hardware": stream.status()})


@csrf_exempt
@require_http_methods(["POST"])
def hardware_buzzer_test(request):
    """Manual Test-buzzer button: beep ~1s on D8 (requires Live serial link)."""
    stream = get_stream()
    if not stream.is_live or not stream.status().get("connected"):
        return JsonResponse(
            {
                "error": "Switch to Live first (serial MPU6050 must be connected).",
                "hardware": stream.status(),
                "buzzer": get_buzzer().status(),
            },
            status=400,
        )
    try:
        body = _parse_body(request)
    except ValueError:
        body = {}
    try:
        seconds = float(body.get("seconds", 1.0))
    except (TypeError, ValueError):
        seconds = 1.0
    seconds = max(0.3, min(seconds, 30.0))
    try:
        status = get_buzzer().pulse_test(seconds=seconds)
    except RuntimeError as exc:
        return JsonResponse(
            {"error": str(exc), "hardware": stream.status(), "buzzer": get_buzzer().status()},
            status=503,
        )
    event_log.log_event("info", "hardware", f"Buzzer TEST ({seconds:.1f}s)")
    return JsonResponse({"ok": True, "buzzer": status, "hardware": stream.status()})


@csrf_exempt
@require_http_methods(["POST"])
def hardware_leds(request):
    try:
        body = _parse_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    leds = get_leds()
    stream = get_stream()
    if "enabled" in body:
        leds.set_enabled(_coerce_bool(body.get("enabled"), True))
    if "auto" in body:
        stream.set_leds_auto(_coerce_bool(body.get("auto"), True))
    if "color" in body:
        try:
            leds.set_color(str(body["color"]).strip().lower())
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
    if "label" in body:
        leds.apply_label(str(body["label"]))

    return JsonResponse({"leds": leds.status(), "hardware": stream.status()})


@csrf_exempt
@require_http_methods(["POST"])
def vision_predict_frame(request):
    """Detect people in a single frame (multipart file or JSON data_url)."""
    try:
        track = _coerce_bool(request.POST.get("track"), False)
        if _coerce_bool(request.POST.get("reset_track"), False):
            vision.reset_tracks()
            ppe.reset_live_votes()
        if request.FILES.get("image"):
            result = vision.predict_image_bytes(request.FILES["image"].read(), track=track)
        else:
            body = _parse_body(request)
            if not body.get("image"):
                return JsonResponse({"error": "Provide image file or data_url in `image`"}, status=400)
            result = vision.predict_data_url(body["image"])
    except Exception as exc:
        event_log.log_event("error", "vision", f"Frame predict failed: {exc}")
        return JsonResponse({"error": str(exc), "torch": vision.get_torch_status()}, status=503)

    event_log.log_event(
        "warning" if result.get("unsafe") else "info",
        "vision",
        f"Vision · {log_line_from_prediction(result)}",
        label=result["label"],
        confidence=result["confidence"],
    )
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def vision_predict_video(request):
    upload = request.FILES.get("video")
    if not upload:
        return JsonResponse({"error": "Upload a video file as `video`"}, status=400)
    max_frames = int(request.POST.get("max_frames", 8))
    video_bytes = upload.read()
    try:
        result = vision.extract_video_frames(video_bytes, max_frames=max_frames)
    except Exception as exc:
        event_log.log_event("error", "vision", f"Video analyze failed: {exc}")
        return JsonResponse({"error": str(exc), "torch": vision.get_torch_status()}, status=400)

    try:
        from .services.recordings import save_analyzed_video

        saved = save_analyzed_video(video_bytes, upload.name)
        result["recording_id"] = saved.pk
        result["recording_url"] = f"/recordings/{saved.pk}/"
    except Exception as exc:
        logger.exception("Could not keep the video log")
        event_log.log_event("error", "vision", f"Video log was not saved: {exc}")

    labels = [
        f["prediction"]["label"]
        for f in result["frames"]
        if f.get("prediction")
    ]
    summary = labels[0] if labels else "unavailable"
    event_log.log_event(
        "info",
        "vision",
        f"Video analyzed ({result['frame_count_sampled']} frames) → {summary}",
        frames=result["frame_count_sampled"],
        torch=result["torch"].get("available"),
    )
    logger.info("vision video frames=%s", result["frame_count_sampled"])
    return JsonResponse(result)


@require_GET
def logs_api(request):
    limit = int(request.GET.get("limit", 100))
    source = request.GET.get("source") or None
    return JsonResponse({"events": event_log.get_events(limit=limit, source=source)})


@csrf_exempt
@require_http_methods(["POST"])
def logs_clear(request):
    event_log.clear_events()
    event_log.log_event("info", "system", "Activity log cleared")
    return JsonResponse({"ok": True})
