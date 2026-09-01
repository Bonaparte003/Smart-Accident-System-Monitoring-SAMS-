#!/usr/bin/env bash
# Generate self-signed TLS certs for phone camera QR (getUserMedia needs HTTPS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$ROOT/hardware/certs"
mkdir -p "$DIR"
IP="$(python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)"
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$DIR/key.pem" \
  -out "$DIR/cert.pem" \
  -days 825 \
  -subj "/CN=SAMS" \
  -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost"
echo "Wrote $DIR/cert.pem for LAN IP $IP"
echo "Phone must accept the certificate warning once, then camera works."
