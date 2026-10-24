"""Time every model on this machine: ``python manage.py benchmark``.

Each stage runs once to warm up, then ``--runs`` times; the median and p90
latency are printed and written to ``results/performance/`` together with the
machine they were measured on, so runs on different hardware can be compared.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from PIL import Image


def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return round(peak / (2**20 if platform.system() == "Darwin" else 2**10), 0)


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def environment() -> dict[str, object]:
    import torch

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": _device(),
    }


def _latest_frame() -> Path | None:
    frames = sorted(Path(settings.MEDIA_ROOT).glob("recordings/frames/**/*.jpg"), key=lambda p: p.stat().st_mtime)
    return frames[-1] if frames else None


class Command(BaseCommand):
    help = "Measure per-model and end-to-end latency on this machine."

    def add_arguments(self, parser):
        parser.add_argument("--image", type=Path, help="Frame to time (default: the newest archived frame)")
        parser.add_argument("--runs", type=int, default=10, help="Timed runs per stage after one warm-up")
        parser.add_argument(
            "--out", type=Path, default=Path(settings.BASE_DIR) / "results" / "performance",
            help="Folder for benchmark.csv and environment.json",
        )

    def handle(self, *args, **opts):
        from monitoring.services import accident_inference, fusion_inference, hazard_inference
        from monitoring.services import machine_state_inference, ppe_inference, vibration_inference
        from monitoring.services import vision_inference

        path = opts["image"] or _latest_frame()
        if path is None or not Path(path).is_file():
            raise CommandError("No frame to time: pass --image path/to/frame.jpg")
        image = Image.open(path).convert("RGB")
        runs = max(1, int(opts["runs"]))
        rows: list[dict[str, object]] = []

        def stage(name: str, fn: Callable[[], object], n: int = runs) -> None:
            fn()
            times = []
            for _ in range(n):
                start = time.perf_counter()
                fn()
                times.append((time.perf_counter() - start) * 1000)
            times.sort()
            row = {
                "stage": name,
                "median_ms": round(statistics.median(times), 1),
                "p90_ms": round(times[max(0, int(0.9 * len(times)) - 1)], 1),
                "runs": n,
            }
            rows.append(row)
            self.stdout.write(f"{name:<28}{row['median_ms']:>10} ms   p90 {row['p90_ms']:>8} ms")

        self.stdout.write(f"Frame {path} ({image.width}x{image.height}), {runs} runs per stage\n")
        if ppe_inference.get_status()["available"]:
            stage("ppe_and_person", lambda: ppe_inference.detect_scene(image))
        pose = vision_inference.get_pose_model()
        if pose is not None:
            stage("pose", lambda: pose.predict(image, verbose=False))
        if hazard_inference.get_model() is not None:
            stage("fire_smoke", lambda: hazard_inference.detect_hazards(image))
        if accident_inference.get_model() is not None:
            stage("fall", lambda: accident_inference.detect_falls(image))
        stage("full_frame", lambda: vision_inference.predict_image(image, annotate=False))
        frame_ms = rows[-1]["median_ms"]

        try:
            row = {col: 0.0 for col in fusion_inference.get_feature_cols()}
            stage("fusion", lambda: fusion_inference.predict_fusion(row), n=runs * 5)
        except (OSError, RuntimeError, KeyError) as exc:
            self.stdout.write(f"fusion skipped: {exc}")
        rng = np.random.default_rng(0)
        for name, module, predict in (
            ("vibration_window", vibration_inference, vibration_inference.predict_from_window),
            ("machine_state_window", machine_state_inference, machine_state_inference.predict_window),
        ):
            size = module.window_size()
            if size:
                window = rng.normal([0.0, 0.0, 1.0], 0.01, (size, 3)).astype(np.float32)
                stage(name, lambda w=window, f=predict: f(w), n=runs * 5)

        env = {**environment(), "peak_rss_mb": _peak_rss_mb(), "frame": str(path),
               "frame_size": [image.width, image.height], "frames_per_s": round(1000 / float(frame_ms), 2)}
        out = Path(opts["out"])
        out.mkdir(parents=True, exist_ok=True)
        with (out / "benchmark.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["stage", "median_ms", "p90_ms", "runs"])
            writer.writeheader()
            writer.writerows(rows)
        (out / "environment.json").write_text(json.dumps(env, indent=2))
        self.stdout.write(
            f"\nFull frame {frame_ms} ms ≈ {env['frames_per_s']} frames/s · peak memory {env['peak_rss_mb']} MB\n"
            f"{env['cpu']} · {env['ram_gb']} GB · {env['device']} · Python {env['python']} · torch {env['torch']}\n"
            f"Saved {out / 'benchmark.csv'} and {out / 'environment.json'}"
        )
