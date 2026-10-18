import io
import json
import logging
import os

import numpy as np
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from PIL import Image

from hardware import get_buzzer, get_camera, get_leds, get_motor, get_stream, get_tof
from hardware.wifi_camera import guess_lan_ip


from .services.interaction_log import record_interaction
from .services.recordings import log_line_from_prediction
from .services.control_center import assemble as assemble_control
from .services.fusion_inference import class_names, get_metadata, get_model
from .services import distance_estimation
from .services import machine_state_inference as machine_state
from .services import trials
from .services.live_recorder import get_recorder
from .services import vibration_inference as vib
from .services import vision_inference as vision

logger = logging.getLogger("monitoring")


def _live_vibration_result():
    """The vibration model's label for the latest MPU6050 window, or None without one."""
    stream = get_stream()
    size = vib.window_size()
    win = stream.latest_window(size) if size else None
    if win is None:
        return None
    result = vib.predict_from_window(win)
    if result is None:
        return None
    result["source"] = "live"
    result["hardware"] = stream.status()
    stream.record_prediction(result)
    return result


@require_http_methods(["GET", "HEAD"])
def favicon(request):
    return HttpResponseRedirect(static("monitoring/favicon.svg") + "?v=plus")


@require_GET
def health(request):
    try:
        get_model()
        meta = get_metadata()
        return JsonResponse(
            {
                "status": "ok",
                "fusion_model": meta.get("experiment_id"),
                "vibration_model": vib.get_status(),
                "machine_state_model": machine_state.get_status(),
                "vision": vision.get_torch_status(),
                "classes": class_names(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("health check failed")
        return JsonResponse({"status": "error", "detail": str(exc)}, status=500)


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


def _control_payload(vision: dict | None, *, actuate: bool) -> dict:
    stream = get_stream()
    vibration = _live_vibration_result() if stream.is_live else None
    machine_size = machine_state.window_size()
    machine = machine_state.predict_window(stream.latest_window(machine_size)) if stream.is_live and machine_size else None
    payload = assemble_control(vision, vibration, machine=machine, actuate=actuate and stream.is_live)
    payload["hardware"] = stream.status()
    payload["tof"] = _tof_status()
    if vibration is not None:
        payload["vibration"]["series"] = vibration.get("series")
    else:
        payload["vibration"] = {
            "label": None,
            "ready": False,
            "detail": "Accelerometer not streaming" if stream.latest_window() is None else "Vibration model unavailable",
        }
    return payload


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
        jpeg = camera.grab_jpeg()
        vision = _live_vision_result(jpeg)
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
    payload["trial"] = trials.record_live_samples(payload, vision)
    try:
        get_recorder().capture(camera.token, jpeg, payload, vision)
    except Exception:  # noqa: BLE001
        logger.exception("Could not keep the live frame in the session recording")
    payload["recording_id"] = get_recorder().recording_id
    decision = (payload.get("fusion") or {}).get("label")
    reason = (payload.get("fusion") or {}).get("reason") or ""
    line = reason or log_line_from_prediction(vision, decision or "")
    message = f"{decision or 'No decision'} · {line}" if line else (decision or "No decision")
    if getattr(control_live, "_last", None) != message:
        control_live._last = message
        level = logging.ERROR if decision == "DANGER" else logging.WARNING if decision == "CAUTION" else logging.INFO
        logger.log(level, "control %s", message)
    payload["ready"] = True
    payload["camera"] = camera.status()
    return JsonResponse(payload)


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
            "motor": get_motor().status(),
            "camera": get_camera().status(),
            "tof": _tof_status(),
        }
    )


@require_GET
def hardware_vibration(request):
    """Raw MPU6050 window for the Live graph — independent of the camera and the model."""
    stream = get_stream()
    win = stream.latest_window()
    if win is None:
        status = stream.status()
        detail = "Accelerometer not streaming" if not status["connected"] else "Filling the first window…"
        return JsonResponse({"ready": False, "detail": detail, "motor": get_motor().status()})
    dev = win - win.mean(axis=0)
    mag = np.sqrt((win.astype(np.float64) ** 2).sum(axis=1))
    return JsonResponse(
        {
            "ready": True,
            "series": {axis: dev[:, i].round(4).tolist() for i, axis in enumerate("xyz")},
            "rms_mg": {axis: round(float(np.sqrt((dev[:, i] ** 2).mean())) * 1000, 1) for i, axis in enumerate("xyz")},
            "mag_std_mg": round(float(mag.std()) * 1000, 1),
            "sample_rate_hz": stream.config.sample_rate_hz,
            "motor": get_motor().status(),
        }
    )


@csrf_exempt
@require_http_methods(["POST", "GET"])
def hardware_motor(request):
    """Run the D9 motor as a simulated machine so the MPU6050 feels real vibration."""
    motor = get_motor()
    if request.method == "GET":
        return JsonResponse({"motor": motor.status()})
    try:
        body = _parse_body(request)
        status = motor.set_level(body.get("level"))
    except ValueError as exc:
        return JsonResponse({"error": str(exc), "motor": motor.status()}, status=400)
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": str(exc), "motor": motor.status()}, status=409)
    return JsonResponse({"ok": True, "motor": status})


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


def _live_vision_result(jpeg: bytes) -> dict:
    """Run every vision model on one phone-pushed frame."""
    camera = get_camera()
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    result: dict = {
        "source": "live",
        "camera": camera.status(),
        "thumbnail": camera.last_thumbnail_data_url(),
    }
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
        get_recorder().finish()
        vision.reset_tracks()
        base = _public_join_base(request)
        status = camera.start_session(join_base=base)
        logger.info("Camera QR session started, waiting for phone")
        return JsonResponse({"ok": True, "camera": status, "join_url": status.get("join_url")})

    if action == "stop" or ("live" in body and not _coerce_bool(body.get("live"), True)):
        get_recorder().finish()
        vision.reset_tracks()
        status = camera.stop_session()
        logger.info("Camera QR session stopped")
        return JsonResponse({"ok": True, "camera": status})

    return JsonResponse({"camera": camera.status()})


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
