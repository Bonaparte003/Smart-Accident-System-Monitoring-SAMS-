from django.contrib import admin

from .models import InteractionEvent, Recording, RecordingMinute


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
