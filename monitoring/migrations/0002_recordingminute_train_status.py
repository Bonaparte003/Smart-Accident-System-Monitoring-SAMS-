from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitoring", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="recordingminute",
            name="train_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("kept", "Kept"),
                    ("excluded", "Excluded"),
                ],
                default="pending",
                max_length=16,
            ),
        ),
    ]
