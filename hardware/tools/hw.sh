#!/usr/bin/env bash
# Terminal workflow for SAMS hardware (Arduino Uno + CH340 on usbserial).
# Usage:
#   ./hardware/tools/hw.sh flash-buzzer   # buzzer-only debug sketch
#   ./hardware/tools/hw.sh flash         # full MPU6050 + buzzer sketch
#   ./hardware/tools/hw.sh test          # send B1/B0, expect ACK
#   ./hardware/tools/hw.sh run           # start Django with serial port set

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${SAMS_HW_SERIAL_PORT:-/dev/cu.usbserial-1410}"
FQBN="${SAMS_HW_FQBN:-arduino:avr:uno}"
cd "$ROOT"

cmd="${1:-help}"

case "$cmd" in
  flash-buzzer)
    echo ">>> Compiling + uploading buzzer_d8_debug → $PORT ($FQBN)"
    arduino-cli compile --fqbn "$FQBN" hardware/firmware/buzzer_d8_debug
    arduino-cli upload -p "$PORT" --fqbn "$FQBN" hardware/firmware/buzzer_d8_debug
    echo ">>> Done. You should hear 3 boot beeps. Run: $0 test"
    ;;
  flash)
    echo ">>> Compiling + uploading mpu6050_serial → $PORT ($FQBN)"
    arduino-cli compile --fqbn "$FQBN" hardware/firmware/mpu6050_serial
    arduino-cli upload -p "$PORT" --fqbn "$FQBN" hardware/firmware/mpu6050_serial
    echo ">>> Done. Run: $0 test   then   $0 run"
    ;;
  test)
    echo ">>> Buzzer serial test on $PORT (stop Django first if it holds the port)"
    # shellcheck disable=SC1091
    source "$ROOT/myenv/bin/activate"
    python hardware/tools/buzzer_test.py --port "$PORT"
    ;;
  run)
    echo ">>> Django HTTPS on 0.0.0.0:8000 (phone camera needs TLS)"
    # shellcheck disable=SC1091
    source "$ROOT/myenv/bin/activate"
    export SAMS_HW_SERIAL_PORT="$PORT"
    export SAMS_HTTPS=1
    CERT="$ROOT/hardware/certs/cert.pem"
    KEY="$ROOT/hardware/certs/key.pem"
    if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
      "$ROOT/hardware/tools/gen_certs.sh"
    fi
    python "$ROOT/hardware/tools/run_https.py" --addr 0.0.0.0 --port 8000 --cert "$CERT" --key "$KEY"
    ;;
  certs)
    "$ROOT/hardware/tools/gen_certs.sh"
    ;;
  ports)
    arduino-cli board list
    ;;
  help|*)
    cat <<EOF
SAMS hardware terminal helper

  export SAMS_HW_SERIAL_PORT=/dev/cu.usbserial-1410   # optional
  ./hardware/tools/hw.sh ports
  ./hardware/tools/hw.sh flash-buzzer
  ./hardware/tools/hw.sh test
  ./hardware/tools/hw.sh flash
  ./hardware/tools/hw.sh test
  ./hardware/tools/hw.sh certs   # regenerate HTTPS cert for current LAN IP
  ./hardware/tools/hw.sh run     # HTTPS Django (required for phone camera QR)

Board FQBN default: arduino:avr:uno  (override with SAMS_HW_FQBN)
EOF
    ;;
esac
