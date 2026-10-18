from django.urls import path

from . import review_views, views

urlpatterns = [
    path("", review_views.entrance, name="review_entrance"),
    path("archive/", review_views.archive, name="review_archive"),
    path("archive/upload/", review_views.upload_video, name="review_upload"),
    path("live/", review_views.live, name="review_live"),
    path("alerts/", review_views.alerts, name="review_alerts"),
    path("mlops/", review_views.mlops, name="review_mlops"),
    path("mlops/label/", review_views.label_minute, name="review_label"),
    path("mlops/export/", review_views.export_training, name="review_export"),
    path("mlops/trials/start/", review_views.start_trial, name="review_trial_start"),
    path("mlops/trials/stop/", review_views.stop_trial, name="review_trial_stop"),
    path("mlops/trials/<int:pk>/delete/", review_views.delete_trial, name="review_trial_delete"),
    path("mlops/trials/export/", review_views.export_trials, name="review_trial_export"),
    path("logout/", review_views.logout_view, name="review_logout"),
    path("recordings/<int:pk>/status/", review_views.recording_status, name="review_recording_status"),
    path("recordings/<int:pk>/", review_views.recording_detail, name="review_recording"),
    path("favicon.ico", views.favicon, name="favicon"),
    path("api/health/", views.health, name="api_health"),
    path("api/control/live/", views.control_live, name="api_control_live"),
    path("api/hardware/status/", views.hardware_status, name="api_hardware_status"),
    path("api/hardware/camera/", views.hardware_camera, name="api_hardware_camera"),
    path("api/hardware/motor/", views.hardware_motor, name="api_hardware_motor"),
    path("api/hardware/vibration/", views.hardware_vibration, name="api_hardware_vibration"),
    path("cam/<str:token>/", views.phone_cam_join, name="phone_cam_join"),
    path("cam/<str:token>/join/", views.phone_cam_join_ack, name="phone_cam_join_ack"),
    path("cam/<str:token>/frame/", views.phone_cam_frame, name="phone_cam_frame"),
]
