"""Payloads and helpers shared by the test modules."""

import io


FUSION_OUT = {
    "label": "CAUTION",
    "label_idx": 1,
    "confidence": 0.71,
    "probabilities": {"NORMAL": 0.1, "CAUTION": 0.71, "DANGER": 0.19},
    "model_id": "fusion_test",
    "feature_cols": [],
}


MEASURED_VISION = {
    "worker_detected": True,
    "confidence": 0.9,
    "worker_distance_est": 0.5,
    "distance_source": "vision",
    "worker_speed": 0.4,
    "approaching": True,
}


VIBRATION = {"label": "NORMAL", "abnormal": False, "anomaly_score": 0.2}


def pose(joints):
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in joints.items():
        points[index] = [float(point[0]), float(point[1]), float(point[2])]
    return points


COLS = ["worker_distance", "vibration_anomaly_score"]


def site_features():
    from monitoring.services.distance_estimation import FEATURE_NAMES

    return {name: 0.5 for name in FEATURE_NAMES}


def live_payload(label):
    return {
        "fusion": {"label": label, "model_id": "fusion_test", "feature_cols": COLS},
        "features": {"worker_distance": 0.6, "vibration_anomaly_score": 0.3},
    }


def jpeg(shade):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (shade, shade, shade)).save(buffer, format="JPEG")
    return buffer.getvalue()


def decided_payload(label):
    return {
        "fusion": {"label": label, "reason": "", "confidence": 0.8, "missing": []},
        "features": {"worker_distance": 0.6, "vibration_anomaly_score": 0.3},
        "vibration": {"label": "NORMAL", "anomaly_score": 0.3, "series": {"x": [0.0]}},
        "machine": {"label": "RUNNING"},
    }


VISION_FRAME = {
    "label": "Person",
    "confidence": 0.9,
    "worker_detected": True,
    "person_count": 1,
    "people": [{"id": 1, "selected": True, "distance_m": 0.6, "distance_source": "vision", "ppe": {}}],
    "detections": [{"xyxy": [1, 2, 30, 40], "track_id": 1, "distance_m": 0.6, "distance_source": "vision"}],
    "frame_w": 64,
    "frame_h": 48,
}
