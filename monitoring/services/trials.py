"""Labelled data-collection trials: live measurements kept under a ground-truth label.

While a trial runs, each live poll where every fusion input was measured is
kept as a fusion row labelled with the trial's risk level, and each person the
ToF beam ranges is kept as a distance row (the sensor's metres label the body
features). The export lays the rows out where notebooks 03 and 04 read them,
so the fusion model can be retrained on real site data instead of synthetic rows.
"""

from __future__ import annotations

import csv
import io
import time
import zipfile
from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

from monitoring.models import RecordingMinute, Trial, TrainingSample

from .distance_estimation import FEATURE_NAMES
from .fusion_inference import class_names, get_feature_cols

FUSION_CSV = "datasets/fusion/fusion_experiment_logs.csv"
SITE_DISTANCE_CSV = "datasets/vision/distance/site_labels.csv"

EXPORT_README = f"""SAMS training data

Unzip at the SAMS folder (MyDrive/SAMS in Colab) so the files land here:

  {FUSION_CSV}
      Fusion rows: every trial row, labelled with its trial's risk level, and
      every Archive moment kept in MLOps, labelled with its reviewed true risk.
      notebooks/03_fusion_risk_classification.ipynb trains on it instead of
      synthetic data. Splits hold out whole trials and recordings, so each
      risk level needs at least three of them. Rows from uploaded clips list
      their operator-declared inputs in the `declared` column.

  {SITE_DISTANCE_CSV}
      Body features of the person in the ToF beam with the sensor's metres.
      notebooks/04_vision_distance_estimation_training.ipynb section 8
      fine-tunes the distance network on it.
"""


def active_trial() -> Trial | None:
    return Trial.objects.filter(ended_at__isnull=True).order_by("-started_at").first()


def _next_code() -> str:
    number = Trial.objects.count() + 1
    while Trial.objects.filter(code=f"T{number:03d}").exists():
        number += 1
    return f"T{number:03d}"


def start_trial(*, risk_label: str, scenario: str = "", participant: str = "", notes: str = "") -> Trial:
    """End any running trial, then start a new one. Its label is the truth for every row it logs."""
    if risk_label not in class_names():
        raise ValueError(f"Risk label must be one of {', '.join(class_names())}")
    end_trial()
    return Trial.objects.create(
        code=_next_code(),
        risk_label=risk_label,
        scenario=scenario.strip()[:64],
        participant=participant.strip()[:64],
        notes=notes.strip(),
    )


def end_trial() -> None:
    Trial.objects.filter(ended_at__isnull=True).update(ended_at=timezone.now())


def with_counts(trials):
    return trials.annotate(
        fusion_rows=Count("samples", filter=Q(samples__kind="fusion")),
        distance_rows=Count("samples", filter=Q(samples__kind="distance")),
    )


def trial_status(trial: Trial | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    counts = dict(trial.samples.values_list("kind").annotate(n=Count("id")))
    return {
        "id": trial.pk,
        "code": trial.code,
        "risk_label": trial.risk_label,
        "scenario": trial.scenario,
        "started_at": trial.started_at.isoformat(),
        "fusion_rows": counts.get("fusion", 0),
        "distance_rows": counts.get("distance", 0),
    }


def record_live_samples(payload: dict[str, Any], vision: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep this poll's rows under the running trial. Returns the trial's status, or None without one.

    A poll with no fusion decision has an unmeasured input, so it gives no fusion row.
    """
    trial = active_trial()
    if trial is None:
        return None
    now_ms = int(time.time() * 1000)
    rows = []
    fusion = payload.get("fusion") or {}
    features = payload.get("features") or {}
    if fusion.get("label"):
        cols = fusion.get("feature_cols") or get_feature_cols()
        rows.append(
            TrainingSample(
                kind="fusion",
                trial=trial,
                timestamp_ms=now_ms,
                features={col: float(features[col]) for col in cols},
                target=trial.risk_label,
                predicted=fusion["label"],
                detail={"model_id": fusion.get("model_id")},
            )
        )
    for pair in (vision or {}).get("distance_pairs") or []:
        rows.append(
            TrainingSample(
                kind="distance",
                trial=trial,
                timestamp_ms=now_ms,
                features=pair["features"],
                target_value=float(pair["distance_m"]),
                detail={"z0_m": float(pair["z0_m"])},
            )
        )
    if rows:
        TrainingSample.objects.bulk_create(rows)
    return trial_status(trial)


def delete_trial(trial: Trial) -> None:
    """Drop a trial and its rows, e.g. when the scene did not match its label."""
    trial.samples.all().delete()
    trial.delete()


FUSION_HEADER = ["trial_id", "trial_code", "scenario", "participant", "source", "declared", "t"]


def _fusion_csv(cols: list[str]) -> str:
    """Trial rows plus the Archive cards kept in MLOps, in the layout notebook 03 reads.

    ``trial_id`` is the group training holds out whole: one trial, or one
    recording. ``declared`` names inputs the operator stated for an uploaded
    clip instead of a sensor measuring them.
    """
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([*FUSION_HEADER, *cols, "label", "label_name", "predicted"])
    samples = (
        TrainingSample.objects.filter(kind="fusion", trial__isnull=False)
        .exclude(target="")
        .select_related("trial")
        .order_by("trial_id", "timestamp_ms")
    )
    step: dict[str, int] = {}

    names = class_names()

    def write(group: str, code: str, scenario: str, participant: str, source: str, declared: str, features, label, predicted):
        if label not in names or any(col not in features for col in cols):
            return
        t = step.get(group, 0)
        step[group] = t + 1
        writer.writerow(
            [group, code, scenario, participant, source, declared, t,
             *[features[col] for col in cols], names.index(label), label, predicted]
        )

    for sample in samples:
        trial = sample.trial
        write(f"trial-{trial.pk}", trial.code, trial.scenario or trial.risk_label, trial.participant,
              "trial", "", sample.features, sample.target, sample.predicted)
    kept = (
        RecordingMinute.objects.filter(train_status="kept")
        .select_related("recording")
        .order_by("recording_id", "minute")
    )
    for mark in kept:
        detail = mark.detail or {}
        recording = mark.recording
        write(f"rec-{recording.pk}", recording.title, recording.source, "", recording.source,
              ";".join(sorted((detail.get("declared") or {}).keys())),
              detail.get("features") or {}, mark.true_label or mark.level, mark.level)
    return out.getvalue()


def _distance_csv() -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["trial_code", *FEATURE_NAMES, "z0_m", "distance_m"])
    samples = TrainingSample.objects.filter(kind="distance").select_related("trial").order_by("timestamp_ms")
    for sample in samples:
        if sample.target_value is None or any(name not in sample.features for name in FEATURE_NAMES):
            continue
        writer.writerow(
            [
                sample.trial.code if sample.trial else "",
                *[sample.features[name] for name in FEATURE_NAMES],
                sample.detail.get("z0_m"),
                sample.target_value,
            ]
        )
    return out.getvalue()


def write_datasets(archive: zipfile.ZipFile) -> None:
    """The CSVs notebooks 03 and 04 read, at their paths under the SAMS folder."""
    archive.writestr(FUSION_CSV, _fusion_csv(get_feature_cols()))
    archive.writestr(SITE_DISTANCE_CSV, _distance_csv())
    archive.writestr("README.txt", EXPORT_README)


def export_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_datasets(archive)
    return buffer.getvalue()
