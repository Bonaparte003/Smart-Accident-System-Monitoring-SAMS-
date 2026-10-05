from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitoring", "0003_interactionevent"),
    ]

    operations = [
        migrations.AddField(
            model_name="recording",
            name="framed",
            field=models.FileField(blank=True, upload_to="recordings/framed/%Y/%m/"),
        ),
        migrations.AddField(
            model_name="recording",
            name="render_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("ready", "Ready"),
                    ("failed", "Failed"),
                ],
                default="pending",
                max_length=16,
            ),
        ),
    ]
