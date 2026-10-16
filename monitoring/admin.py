from django.contrib import admin

from .models import InteractionEvent, Recording, RecordingMinute, Trial, TrainingSample


class RecordingMinuteInline(admin.TabularInline):
    model = RecordingMinute
    extra = 0
    fields = ("minute", "level", "summary", "offset_s", "train_status")
    readonly_fields = fields


@admin.register(InteractionEvent)
class InteractionEventAdmin(admin.ModelAdmin):
    list_display = ("created_at", "level", "reason", "person_id", "distance_m", "machine_on")
    readonly_fields = ("created_at",)


@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ("title", "created_at", "duration_s", "problem_count", "render_status")
    inlines = [RecordingMinuteInline]


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ("code", "started_at", "ended_at", "risk_label", "scenario", "participant")


@admin.register(TrainingSample)
class TrainingSampleAdmin(admin.ModelAdmin):
    list_display = ("kind", "trial", "timestamp_ms", "target", "target_value", "predicted")
    list_filter = ("kind",)
