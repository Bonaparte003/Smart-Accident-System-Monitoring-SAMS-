from django.db import models


class Recording(models.Model):
    """A workplace video kept so a problem minute can be opened later."""

    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=200)
    video = models.FileField(upload_to="recordings/%Y/%m/")
    framed = models.FileField(upload_to="recordings/framed/%Y/%m/", blank=True)
    render_status = models.CharField(
        max_length=16,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("ready", "Ready"),
            ("failed", "Failed"),
        ],
    )
    duration_s = models.FloatField(default=0)
    fps = models.FloatField(default=0)
    problem_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    @property
    def duration_label(self) -> str:
        total = max(0, int(round(self.duration_s)))
        minutes, seconds = divmod(total, 60)
        return f"{minutes}:{seconds:02d}"


class RecordingMinute(models.Model):
    """One minute of a recording, with the decision at that point in the video."""

    recording = models.ForeignKey(Recording, related_name="minutes", on_delete=models.CASCADE)
    minute = models.PositiveIntegerField()
    offset_s = models.FloatField(default=0)
    level = models.CharField(max_length=16, default="NORMAL")
    summary = models.CharField(max_length=240, blank=True)
    snapshot = models.ImageField(upload_to="recordings/frames/%Y/%m/", blank=True)
    detail = models.JSONField(default=dict)
    train_status = models.CharField(
        max_length=16,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("kept", "Kept"),
            ("excluded", "Excluded"),
        ],
    )

    class Meta:
        ordering = ["minute"]
        constraints = [
            models.UniqueConstraint(fields=["recording", "minute"], name="unique_recording_minute"),
        ]

    @property
    def problem(self) -> bool:
        return self.level in {"CAUTION", "DANGER", "DEGRADED"}

    @property
    def clock(self) -> str:
        minutes, seconds = divmod(int(self.offset_s), 60)
        return f"{minutes}:{seconds:02d}"


class InteractionEvent(models.Model):
    """One live decision, kept when the class, sentence, person, or machine switch changes."""

    created_at = models.DateTimeField(auto_now_add=True)
    person_id = models.CharField(max_length=32, blank=True)
    level = models.CharField(max_length=16, default="NORMAL")
    reason = models.CharField(max_length=240, blank=True)
    distance_m = models.FloatField(null=True, blank=True)
    distance_source = models.CharField(max_length=16, blank=True)
    machine_on = models.BooleanField(default=False)
    vibration_score = models.FloatField(null=True, blank=True)
    vibration_unusual = models.BooleanField(default=False)
    approaching = models.BooleanField(default=False)
    detail = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.level} {self.reason}".strip()


RISK_LABELS = [("NORMAL", "Normal"), ("CAUTION", "Caution"), ("DANGER", "Danger")]
VIBRATION_LABELS = [("NORMAL", "Normal"), ("ABNORMAL", "Abnormal")]


class Trial(models.Model):
    """One scripted data-collection run. Its labels are the ground truth for its samples."""

    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    code = models.CharField(max_length=32)
    participant = models.CharField(max_length=64, blank=True)
    scenario = models.CharField(max_length=64, blank=True)
    risk_label = models.CharField(max_length=16, choices=RISK_LABELS, blank=True)
    vibration_label = models.CharField(max_length=16, choices=VIBRATION_LABELS, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return self.code

    @property
    def active(self) -> bool:
        return self.ended_at is None


class TrainingSample(models.Model):
    """A feature row logged live for retraining a model on this site's data."""

    KINDS = [
        ("fusion", "Fusion window"),
        ("distance", "Sensor / body distance pair"),
        ("vibration", "Accelerometer window"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    kind = models.CharField(max_length=16, choices=KINDS)
    trial = models.ForeignKey(Trial, null=True, blank=True, related_name="samples", on_delete=models.SET_NULL)
    timestamp_ms = models.BigIntegerField()
    features = models.JSONField(default=dict)
    target = models.CharField(max_length=16, blank=True)
    target_value = models.FloatField(null=True, blank=True)
    predicted = models.CharField(max_length=16, blank=True)
    detail = models.JSONField(default=dict)

    class Meta:
        ordering = ["timestamp_ms"]
        indexes = [models.Index(fields=["kind", "timestamp_ms"])]
