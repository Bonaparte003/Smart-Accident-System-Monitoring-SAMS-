from django.apps import AppConfig


class MonitoringConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "monitoring"
    verbose_name = "SAMS Monitoring"

    def ready(self) -> None:
        import sys

        command = " ".join(sys.argv)
        if any(token in command for token in ("test", "migrate", "makemigrations", "shell", "collectstatic")):
            return
        from monitoring.services.video_jobs import resume_pending

        resume_pending()
