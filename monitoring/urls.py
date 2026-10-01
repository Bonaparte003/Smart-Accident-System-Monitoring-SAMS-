from django.urls import path

from . import views

urlpatterns = [
    path("", views.app, name="app"),
    path("favicon.ico", views.favicon, name="favicon"),
    path("api/health/", views.health, name="api_health"),
    path("api/config/", views.config, name="api_config"),
    path("api/simulate/", views.simulate, name="api_simulate"),
    path("api/fusion/predict/", views.predict_fusion_api, name="api_fusion_predict"),
    path("api/vibration/predict/", views.vibration_predict, name="api_vibration_predict"),
    path("api/vibration/live/", views.vibration_live, name="api_vibration_live"),
    path("api/hardware/status/", views.hardware_status, name="api_hardware_status"),
    path("api/hardware/mode/", views.hardware_mode, name="api_hardware_mode"),
    path("api/hardware/buzzer/", views.hardware_buzzer, name="api_hardware_buzzer"),
    path("api/hardware/buzzer/test/", views.hardware_buzzer_test, name="api_hardware_buzzer_test"),
    path("api/hardware/leds/", views.hardware_leds, name="api_hardware_leds"),
    path("api/hardware/camera/", views.hardware_camera, name="api_hardware_camera"),
    path("cam/<str:token>/", views.phone_cam_join, name="phone_cam_join"),
    path("cam/<str:token>/join/", views.phone_cam_join_ack, name="phone_cam_join_ack"),
    path("cam/<str:token>/frame/", views.phone_cam_frame, name="phone_cam_frame"),
    path("api/vision/predict/", views.vision_predict_frame, name="api_vision_predict"),
    path("api/vision/video/", views.vision_predict_video, name="api_vision_video"),
    path("api/vision/live/", views.vision_live, name="api_vision_live"),
    path("api/logs/", views.logs_api, name="api_logs"),
    path("api/logs/clear/", views.logs_clear, name="api_logs_clear"),
]
