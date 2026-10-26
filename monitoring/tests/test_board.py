"""Board-side models and hardware: vibration, machine state, ToF, serial port, motor."""

from unittest.mock import patch

from django.test import SimpleTestCase, TestCase


class VibrationModelTests(SimpleTestCase):
    def test_the_trained_model_labels_the_window(self):
        import numpy as np

        from monitoring.services import vibration_inference as vib

        class Model:
            classes_ = np.array([0, 1])

            def predict_proba(self, rows):
                return np.array([[0.3, 0.7]] * len(rows))

        window = np.zeros((100, 3), dtype=np.float32)
        window[:, 2] = 1.0
        meta = {
            "experiment_id": "vib_test",
            "class_names": ["NORMAL", "ABNORMAL"],
            "sample_rate_hz": 50.0,
            "window_size": 100,
            "feature_cols": list(vib.window_to_features(window, 50.0)),
        }
        with patch.object(vib, "get_model", return_value=Model()), patch.object(vib, "get_metadata", return_value=meta):
            self.assertIsNone(vib.predict_from_window(window[:60]))
            result = vib.predict_from_window(window)
        self.assertEqual(result["label"], "ABNORMAL")
        self.assertEqual(result["anomaly_score"], 0.7)
        self.assertEqual(result["model_id"], "vib_test")
        with patch.object(vib, "get_model", return_value=None):
            self.assertIsNone(vib.predict_from_window(window))

    def test_a_flat_window_has_no_nan_features(self):
        import math

        import numpy as np

        from monitoring.services.vibration_inference import window_to_features

        flat = np.zeros((100, 3), dtype=np.float32)
        flat[:, 2] = 1.0
        self.assertTrue(all(math.isfinite(value) for value in window_to_features(flat, 50.0).values()))


class ToFPresenceTests(SimpleTestCase):
    def test_tof_is_online_only_while_the_board_reports_it(self):
        from hardware.presence import get_presence
        from hardware.tof import get_tof

        presence = get_presence()
        presence.reset()
        try:
            self.assertFalse(get_tof().online())
            presence.note_line("# TOF raw=8190 timeout=0 status=4")
            self.assertTrue(get_tof().online())
            presence.note_line("# TOF not found model=0xFF")
            self.assertFalse(get_tof().online())
        finally:
            presence.reset()


class SerialPortTests(SimpleTestCase):
    def test_an_explicit_port_is_used(self):
        import os

        from hardware.bringup import find_serial_port

        previous = os.environ.get("SAMS_HW_SERIAL_PORT")
        os.environ["SAMS_HW_SERIAL_PORT"] = "/dev/cu.usbserial-test"
        try:
            self.assertEqual(find_serial_port(), "/dev/cu.usbserial-test")
        finally:
            if previous is None:
                os.environ.pop("SAMS_HW_SERIAL_PORT", None)
            else:
                os.environ["SAMS_HW_SERIAL_PORT"] = previous


class HardwareHealthTests(TestCase):
    def _rows(self):
        from monitoring.review_views import _health

        return {row["label"]: row for row in _health()}

    def test_absent_sensors_stay_inactive(self):
        import os

        from hardware import get_stream
        from hardware.presence import get_presence

        stream = get_stream()
        previous = stream._source
        stream._source = None
        get_presence().reset()
        os.environ["SAMS_HW_TOF_M"] = "1.2"
        try:
            rows = self._rows()
        finally:
            stream._source = previous
            get_presence().reset()
            os.environ.pop("SAMS_HW_TOF_M", None)
        for name in ("Phone camera", "Accelerometer", "Distance sensor", "Buzzer"):
            self.assertFalse(rows[name]["ok"], name)
            self.assertEqual(rows[name]["status"], "Not connected")

    def test_only_a_sensor_the_board_reports_is_active(self):
        from hardware import get_stream
        from hardware.presence import get_presence

        class _Open:
            connected = True

        stream = get_stream()
        previous = stream._source
        stream._source = _Open()
        presence = get_presence()
        presence.reset()
        presence.note_sample()
        presence.note_line("# ACK B1")
        presence.note_line("# TOF not found")
        try:
            rows = self._rows()
        finally:
            stream._source = previous
            presence.reset()
        self.assertTrue(rows["Accelerometer"]["ok"])
        self.assertTrue(rows["Buzzer"]["ok"])
        self.assertFalse(rows["Distance sensor"]["ok"])
        self.assertFalse(rows["Phone camera"]["ok"])
        self.assertEqual(rows["Distance sensor"]["status"], "Not connected")


class MachineMotorTests(SimpleTestCase):
    def setUp(self):
        from hardware.motor import Motor

        self.sent: list[str] = []
        self.motor = Motor()
        patcher = patch("monitoring.views.get_motor", return_value=self.motor)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, level):
        return self.client.post("/api/hardware/motor/", {"level": level}, content_type="application/json")

    def _acking_writer(self, cmd):
        self.sent.append(cmd)
        self.motor.note_ack(f"# ACK {cmd}")

    def test_levels_send_serial_commands(self):
        self.motor.attach_writer(self._acking_writer)
        for level in ("low", "medium", "high", "intermittent", "off"):
            self.assertEqual(self._post(level).status_code, 200, level)
        self.assertEqual(self.sent, ["M120", "M180", "M255", "V1", "V0"])
        self.assertEqual(self.motor.status()["level"], "off")
        self.assertTrue(self.motor.status()["acked"])

    def test_old_firmware_shows_as_unacknowledged(self):
        self.motor.ACK_TIMEOUT_S = 0.05
        self.motor.attach_writer(self.sent.append)  # never answers, like firmware without M<duty>
        self.assertEqual(self._post("low").status_code, 200)
        self.assertFalse(self.motor.status()["acked"])

    def test_needs_the_board(self):
        res = self._post("high")
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.json()["motor"]["level"], "off")

    def test_rejects_unknown_level(self):
        self.motor.attach_writer(self.sent.append)
        self.assertEqual(self._post("warp").status_code, 400)
        self.assertEqual(self.sent, [])

    def test_stream_stop_turns_the_machine_off(self):
        self.motor.attach_writer(self.sent.append)
        self.motor.set_level("high")
        self.motor.detach()
        self.assertEqual(self.sent[-1], "V0")
        self.assertFalse(self.motor.status()["linked"])

    def test_vibration_feed_is_centred_per_axis(self):
        import numpy as np

        from hardware import get_stream

        win = np.zeros((100, 3), dtype=np.float32)
        win[:, 2] = 1.0
        win[::2, 0] = 0.1  # x swings 0 / 0.1 g
        with patch.object(get_stream(), "latest_window", return_value=win):
            data = self.client.get("/api/hardware/vibration/").json()
        self.assertTrue(data["ready"])
        self.assertAlmostEqual(data["series"]["z"][0], 0.0)
        self.assertAlmostEqual(data["rms_mg"]["x"], 50.0, places=0)
        self.assertEqual(data["rms_mg"]["z"], 0.0)
