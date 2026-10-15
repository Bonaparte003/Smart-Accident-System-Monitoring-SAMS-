#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${SAMS_HW_SERIAL_PORT:-$(ls /dev/cu.usbserial-* /dev/cu.wchusbserial* 2>/dev/null | head -n 1 || true)}"
PORT="${PORT:-/dev/cu.usbserial-1410}"
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
  flash-motor)
    echo ">>> Compiling + uploading motor_d9_test → $PORT ($FQBN)"
    arduino-cli compile --fqbn "$FQBN" hardware/firmware/motor_d9_test
    arduino-cli upload -p "$PORT" --fqbn "$FQBN" hardware/firmware/motor_d9_test
    echo ">>> Done. Motor(s) on D9 should pulse 3 times. Drive it with:"
    echo "    arduino-cli monitor -p $PORT -c baudrate=115200   (type 1, 0, M128, S, P)"
    ;;
  flash-scan)
    echo ">>> Compiling + uploading i2c_pin_scan → $PORT ($FQBN)"
    arduino-cli compile --fqbn "$FQBN" hardware/firmware/i2c_pin_scan
    arduino-cli upload -p "$PORT" --fqbn "$FQBN" hardware/firmware/i2c_pin_scan
    echo ">>> Done. Read it with: arduino-cli monitor -p $PORT -c baudrate=115200"
    ;;
  flash-mpu-diag)
    echo ">>> Compiling + uploading mpu_diag → $PORT ($FQBN)"
    arduino-cli compile --fqbn "$FQBN" hardware/firmware/mpu_diag
    arduino-cli upload -p "$PORT" --fqbn "$FQBN" hardware/firmware/mpu_diag
    echo ">>> Done. Read it with: arduino-cli monitor -p $PORT -c baudrate=115200"
    ;;
  test)
    echo ">>> Buzzer serial test on $PORT (stop Django first if it holds the port)"
    source "$ROOT/myenv/bin/activate"
    python hardware/tools/buzzer_test.py --port "$PORT"
    ;;
  monitor)
    if lsof "$PORT" >/dev/null 2>&1; then
      echo ">>> $PORT is held by another process (Django?). Stop it first —"
      echo "    while the server runs, the board's # lines show in its log as 'MCU RX …'."
      exit 1
    fi
    echo ">>> Serial monitor on $PORT @ 115200 — type commands (B1, LG, M180, V0, ST…), Ctrl+C to quit"
    arduino-cli monitor -p "$PORT" -c baudrate=115200
    ;;
  check)
    echo ">>> Full hardware check on $PORT (stop Django first if it holds the port)"
    source "$ROOT/myenv/bin/activate"
    shift
    python hardware/tools/hw_check.py --port "$PORT" "$@"
    ;;
  test-leds)
    echo ">>> LED cycle on $PORT (green D2, yellow D3, red D5)"
    source "$ROOT/myenv/bin/activate"
    python hardware/tools/led_test.py --port "$PORT"
    ;;
  record-steady)
    echo ">>> Recording the MPU6050 on a stopped machine from $PORT (stop Django first if it holds the port)"
    source "$ROOT/myenv/bin/activate"
    shift
    python hardware/tools/record_steady.py --port "$PORT" "$@"
    ;;
  record-running)
    echo ">>> Recording the MPU6050 on a running machine from $PORT (stop Django first if it holds the port)"
    source "$ROOT/myenv/bin/activate"
    shift
    python hardware/tools/record_steady.py --port "$PORT" --state RUNNING "$@"
    ;;
  run)
    echo ">>> Uvicorn on port 8000 (HTTPS; phone camera needs it)"
    source "$ROOT/myenv/bin/activate"
    export SAMS_HW_SERIAL_PORT="$PORT"
    export SAMS_HTTPS=1
    python "$ROOT/manage.py" runserver 0.0.0.0:8000
    ;;
  certs)
    "$ROOT/hardware/tools/gen_certs.sh"
    ;;
  ports)
    arduino-cli board list
    ;;
  port)
    echo "$PORT"
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
  ./hardware/tools/hw.sh flash-motor # motor(s) on D9: 3 pulses, then 1/0/M<0-255>/S/P over serial
  ./hardware/tools/hw.sh flash-scan  # find the MPU6050 on any free pin pair
  ./hardware/tools/hw.sh flash-mpu-diag  # line voltages + slow WHO_AM_I on A2/A3
  ./hardware/tools/hw.sh monitor     # watch what the board prints and type commands to it
  ./hardware/tools/hw.sh check       # banner, MPU rate/values, ToF, buzzer/LED/ST ACKs
  ./hardware/tools/hw.sh test-leds   # green, yellow, red, then off
  ./hardware/tools/hw.sh record-steady --name press01    # machine off: STOPPED data for notebook 02
  ./hardware/tools/hw.sh record-running --name press01   # machine on: RUNNING data from the same board
                                                          # drives the D9 motor (--motor low|medium|high|intermittent|none)
  ./hardware/tools/hw.sh certs   # regenerate HTTPS cert for current LAN IP
  ./hardware/tools/hw.sh run     # uvicorn HTTPS (phone camera needs the https URL)

Board FQBN default: arduino:avr:uno  (override with SAMS_HW_FQBN)
EOF
    ;;
esac
