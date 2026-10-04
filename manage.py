#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import subprocess
import sys
from pathlib import Path


def _runserver_bind(argv):
    """Address and port from `manage.py runserver [addr:port]`. Default listens on the LAN."""
    addr, port = "0.0.0.0", 8000
    noreload = False
    for arg in argv[2:]:
        if arg == "--noreload":
            noreload = True
            continue
        if arg.startswith("-"):
            continue
        if arg.isdigit():
            port = int(arg)
            continue
        host, sep, raw_port = arg.rpartition(":")
        if sep and raw_port.isdigit():
            addr = host or "0.0.0.0"
            port = int(raw_port)
    return addr, port, noreload


def _launch_uvicorn(argv):
    """runserver opens the board and serves HTTPS with uvicorn."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SAMS.settings")
    from hardware.bringup import ensure_certs, prepare_environment
    from hardware.wifi_camera import guess_lan_ip

    prepare_environment()
    addr, port, noreload = _runserver_bind(argv)
    try:
        cert, key = ensure_certs()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not prepare the phone-camera certificate: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "SAMS.asgi:application",
        "--host",
        addr,
        "--port",
        str(port),
        "--ssl-certfile",
        str(cert),
        "--ssl-keyfile",
        str(key),
    ]
    if not noreload:
        command += [
            "--reload",
            "--reload-dir",
            str(root / "SAMS"),
            "--reload-dir",
            str(root / "monitoring"),
            "--reload-dir",
            str(root / "hardware"),
        ]

    lan = guess_lan_ip()
    print("SAMS is starting with uvicorn (HTTPS):", flush=True)
    print(f"  Mac:   https://127.0.0.1:{port}/", flush=True)
    if lan and not lan.startswith("127."):
        print(f"  Phone: https://{lan}:{port}/  (camera needs HTTPS)", flush=True)
    print("The certificate is self-signed. In the browser choose Advanced, then continue.", flush=True)
    os.execv(sys.executable, command)


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SAMS.settings")
    if len(sys.argv) > 1 and sys.argv[1] == "runserver" and not any(
        flag in sys.argv for flag in ("-h", "--help")
    ):
        _launch_uvicorn(sys.argv)
        return
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
