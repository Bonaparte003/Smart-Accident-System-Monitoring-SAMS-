# SAMS hardware

Physical sensors and actuators for **live** (non-simulation) monitoring.

## What you need now

| Piece | Required? | Role |
| --- | --- | --- |
| MPU6050 | Yes | Live accelerometer → vibration model |
| Buzzer on **D8** | Yes (for alerts) | Beeps on caution / danger |
| Android phone (Wi‑Fi) | Yes (for vision) | Live camera → YOLOv8n person + distance NN |
| Time-of-flight (UART `TOF:m`) | Optional (near range) | Close-range metres; vision NN covers far |
| LEDs | **Optional** | Green / yellow / red — skip until you have them |

### Wiring (Arduino / Nano)

See [`firmware/mpu6050_serial.ino`](firmware/mpu6050_serial.ino).

**Minimum**

| Device | Pin |
| --- | --- |
| MPU6050 SDA / SCL | A4 / A5 (Uno) |
| VL53L0X SDA / SCL | **A0 / A1** |
| VL53L0X VIN / GND | 5V / GND. Leave XSHUT open. |
| Buzzer | **D8** and **GND** (`+`→D8, `−`→GND for active; or VCC/GND/I/O module). If it sings on 3V but not on D8, use a transistor so D8 only switches the rail. |

**LED Setup**

| LED | Pin |
| --- | --- |
| Green | D2 |
| Yellow | D3 |
| Red | D5 |

Re-flash after updates. On upload/reset you should hear a **double chirp** and
see the onboard LED (D13) follow the buzzer when commands arrive.

### Android phone camera (QR join)

Phone browsers **block the camera on plain HTTP**. Run Django over HTTPS:

```bash
./hardware/tools/hw.sh certs   # once / after Wi‑Fi IP changes
./hardware/tools/hw.sh run     # uvicorn https://0.0.0.0:8000
```

Then open the Mac UI at `https://192.168.x.x:8000` (accept the cert warning),
go to **Vision lab → Live phone**, and scan the QR with the phone.

On the phone: accept the certificate warning → **Allow camera**. Frames stream
to the Mac; YOLOv8n detects people live (bounding boxes + `worker_detected`).
Distance uses a residual pose-geometry NN for **far** range; inject near-range
with `SAMS_HW_TOF_M=1.2` or firmware lines `TOF:1.23` (see
[`models/vision/distance/README.md`](../models/vision/distance/README.md)).

If the QR still shows `http://…`, restart with `hw.sh run` (sets `SAMS_HTTPS=1`).
Override explicitly with:

```bash
export SAMS_PUBLIC_BASE=https://192.168.1.198:8000
```

### Terminal workflow (recommended)

Your board on `cu.usbserial-*` is an **Arduino Uno / AVR + CH340**. Upload with
`arduino-cli` as **Uno** — do not use Board “Nano 33 BLE” for this port.

```bash
cd /Users/avellin/Desktop/Campus/SAMS

export SAMS_HW_SERIAL_PORT=/dev/cu.usbserial-1410

./hardware/tools/hw.sh flash-buzzer   # buzzer-only test sketch
./hardware/tools/hw.sh test           # expect: RX # ACK B1

./hardware/tools/hw.sh flash          # full MPU6050 + buzzer
./hardware/tools/hw.sh test
./hardware/tools/hw.sh run            # uvicorn with the serial port set
```

Then in the UI: **Vibration lab → Live → Test buzzer**. Server log should show
`MCU TX B1` and `MCU RX # ACK B1`.

If ACK works but there is no sound: D8 cannot power the buzzer — use a transistor
(or 3‑pin module with VCC to 5V). Active buzzers beep on DC HIGH; passive piezos
need `tone()` (not used here — firmware drives a solid HIGH on B1/B2).

### Serial protocol (115200)

**MCU → host** (~50 Hz): `ax,ay,az`

**Host → MCU**

| Command | Effect |
| --- | --- |
| `B0` / `B1` / `B2` | Buzzer off / caution / danger |
| `L0` / `LG` / `LY` / `LR` | LEDs (optional) |

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SAMS_HW_SERIAL_PORT` | _(required for Live vib)_ | e.g. `/dev/cu.usbserial-1410` |
| `SAMS_HW_SERIAL_BAUD` | `115200` | UART baud |
| `SAMS_HW_BUZZER_AUTO` | `true` | Drive buzzer from live risk labels |
| `SAMS_HW_BUZZER_ENABLED` | `true` | Master buzzer enable |
| `SAMS_HW_LEDS_AUTO` | `true` | Green on normal; yellow / red on risk |
| `SAMS_HW_LEDS_ENABLED` | `true` | Master LED enable |
| `SAMS_PUBLIC_BASE` | _(auto LAN IP)_ | Base URL encoded in the phone QR, e.g. `http://192.168.1.20:8000` |
| `SAMS_PUBLIC_HOST` | | Optional host/IP override used when building the QR |
| `SAMS_VISION_POSE` | `false` | Optional YOLOv8n-pose keypoints for distance features |
| `SAMS_VISION_PPE` | `true` | Live hard-hat / vest / goggles / gloves detector |
| `SAMS_PPE_WEIGHTS` | `models/vision/ppe/best.pt` | Override PPE weights path |
| `SAMS_HW_TOF_M` | | Inject ToF metres (lab without sensor) |
| `SAMS_HW_TOF_ENABLED` | `true` | Master ToF enable |

```bash
export SAMS_HW_SERIAL_PORT=/dev/cu.usbserial-1410
./hardware/tools/hw.sh run
# open https://<your-lan-ip>:8000 on the browser
```

Then: **Vibration lab → Live → Buzz caution** and **Vision lab → Live phone** (phone scans QR).

### API

- `GET /api/hardware/status/`
- `POST /api/hardware/mode/` — `{ "mode": "live"|"simulate" }`
- `POST /api/hardware/buzzer/` — `{ "state": "off"|"caution"|"danger" }`
- `POST /api/hardware/leds/` — optional
- `GET|POST /api/hardware/camera/` — `{ "action": "start"|"stop" }` (QR session)
- `GET /cam/<token>/` — phone join page (from QR)
- `POST /cam/<token>/frame/` — phone uploads JPEG
- `GET /api/vibration/live/`
- `GET /api/vision/live/` — classify latest phone frame
