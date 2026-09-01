#!/usr/bin/env python3
"""Run Django over HTTPS (phone getUserMedia requires a secure context on LAN)."""

from __future__ import annotations

import argparse
import os
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SAMS.settings")
os.environ.setdefault("SAMS_HTTPS", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cert",
        default=str(ROOT / "hardware" / "certs" / "cert.pem"),
    )
    parser.add_argument(
        "--key",
        default=str(ROOT / "hardware" / "certs" / "key.pem"),
    )
    args = parser.parse_args()

    cert = Path(args.cert)
    key = Path(args.key)
    if not cert.is_file() or not key.is_file():
        print(f"Missing TLS files. Run: {ROOT}/hardware/tools/gen_certs.sh", file=sys.stderr)
        return 1

    import django

    django.setup()

    from django.contrib.staticfiles.handlers import StaticFilesHandler
    from django.core.servers.basehttp import WSGIRequestHandler, WSGIServer
    from django.core.wsgi import get_wsgi_application

    class QuietHandler(WSGIRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    class DualStackServer(WSGIServer):
        allow_reuse_address = True
        daemon_threads = True

    application = StaticFilesHandler(get_wsgi_application())

    httpd = DualStackServer((args.addr, args.port), QuietHandler)
    httpd.set_app(application)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"SAMS HTTPS at https://{args.addr}:{args.port}/")
    print("Open the LAN IP in the Mac browser, accept the cert, then scan QR with the phone.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
