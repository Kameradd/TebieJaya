# TebieJaya

> Greatness comes with a price

![Python](https://img.shields.io/badge/Python-81.4%25-blue)
![C++](https://img.shields.io/badge/C++-18.6%25-orange)
![Status](https://img.shields.io/badge/status-WIP-lightgrey)

---

## Features
TBA

---

### 1. ESP32/

Contains Arduino sketches for sensor acquisition and MQTT publishing:

Subfolders / Sketches:
- `ESP32_DHT22_MAX4466_PIR/` – Reads DHT22 (temp/humidity), MAX4466 microphone (RMS proxy), PIR motion; streams over UART (Serial2).
- `ESP32_DHT22_MAX4466_PIR_MQTT/` – Same sensors + publishes JSON payload via secure MQTT (TLS) to public broker.
- `ESP32_MQTT_DEFAULT/` – Minimal MQTT over TLS connectivity template.
- `ESP32_MQTT_TEST/` – TLS connection test & message publishing loop.

#### Hardware Prerequisites
- ESP32 development board (e.g. ESP32-DevKitC, DOIT, etc.)
- Sensors:
  - DHT22 on GPIO 4 (with 10k pull‑up between VCC and DATA if needed)
  - PIR sensor on GPIO 5
  - MAX4466 analog mic → ADC pin GPIO 34 (ensure 3.3V supply)
- (Optional) Logic-level compatible USB–serial cable if direct debugging beyond onboard USB.
- For UART link to Jetson: use ESP32 `Serial2` pins in code (RX=16, TX=17) → Jetson UART (3.3V TTL). GND MUST be common.

#### Software / Tooling
| Item | Notes |
|------|-------|
| Arduino IDE ≥ 2.x OR PlatformIO | Either is fine |
| ESP32 Board Package | Install via Boards Manager: `https://espressif.github.io/arduino-esp32/package_esp32_index.json` |
| Python (optional) | Only if using PlatformIO or build scripts |

#### Arduino Libraries (install via Library Manager)
| Library | Purpose | Example Name in Library Manager |
|---------|---------|----------------------------------|
| `DHT sensor library` | DHT22 temperature/humidity | Adafruit DHT sensor library |
| `Adafruit Unified Sensor` | DHT dependency | Adafruit Unified Sensor |
| `PubSubClient` | MQTT client | PubSubClient |
| (Built-in) `WiFi.h` | Wi-Fi stack | Part of ESP32 core |
| (Built-in) `time.h` | NTP/time sync | Part of toolchain |
| (TLS) BearSSL | Included in ESP32 core when using `WiFiClientSecure` |

#### Secure MQTT (TLS)
In the TLS sketches you embed a root CA certificate:
- Keep CA blocks (`-----BEGIN CERTIFICATE----- ...`) intact.
- Replace broker, username, password env-style (do NOT hardcode in production).
- For EMQX or custom broker: download appropriate root CA.

#### Configuration Fields to Change
| Symbol / Variable | Replace With |
|-------------------|-------------|
| `ssid`, `password` | Your WiFi credentials |
| `mqtt_server` / `mqtt_broker` | Broker hostname or IP |
| `mqtt_port` | `8883` for TLS, `1883` for plain |
| `mqtt_username`, `mqtt_password` | Broker auth credentials |
| `mqtt_topic` | esp32 |
| `ca_cert` | Proper root CA for your broker |
| `DB_THRESHOLD` | Calibrate mic threshold (start ~45–55) |

#### Build & Flash (Arduino CLI example)
```bash
arduino-cli board list
arduino-cli core install esp32:esp32
arduino-cli lib install "PubSubClient" "DHT sensor library" "Adafruit Unified Sensor"

arduino-cli compile --fqbn esp32:esp32:esp32 ESP32/ESP32_DHT22_MAX4466_PIR
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 ESP32/ESP32_DHT22_MAX4466_PIR
```

#### PlatformIO (alternative)
Create `platformio.ini` (example):
```ini
[env:esp32]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
  adafruit/DHT sensor library
  adafruit/Adafruit Unified Sensor
  knolleary/PubSubClient
```

---

### 2. LocalPC/

Computer vision + tracking + YOLO + ReID pipelines.

Key Scripts:
- `PY_DEPLOY_EZVIZ.py` / `PY_DEPLOY_EZVIZ2.py` / `PY_DEPLOY_EZVIZ3*.py` / `PY_DEPLOY_EZVIZ4.py` – Progressive iterations of live RTSP goat tracking with posture heuristic & temperature proxy.
- `PY_TEST_EZVIZ.py` – Simple RTSP connectivity test.
- `CHECK_GPU.py` – CUDA availability report.
- `best.pt` – YOLO model weights (Ultralytics format).
- `boxmot/reid_weights/osnet_x0_25_msmt17.pt` – ReID weights for StrongSORT (BoxMOT).
- Generated outputs: `goat_tracking_output_live.mp4`, `goat_tracks_live.csv`.

#### System Prerequisites
| Component | Recommendation |
|-----------|----------------|
| Python | 3.10–3.11 (Ultralytics + Torch stable) |
| GPU (optional) | NVIDIA GPU + latest driver + CUDA toolkit (or use PyTorch wheels w/ bundled CUDA) |
| FFmpeg | Required for OpenCV RTSP low-latency (`ffmpeg` in PATH) |
| OS | Linux recommended (Windows works but adjust paths) |

Install FFmpeg:
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install -y ffmpeg
```

#### Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel setuptools
```

#### Core Python Dependencies
| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLO detection |
| `torch` / `torchvision` | Deep learning backend |
| `opencv-python` (or `opencv-python-headless`) | Video + image ops |
| `numpy` | Array math |
| `pandas` | CSV logging |
| `boxmot` | Multi-object tracking (StrongSORT) |
| `scipy` (optional) | Might be required by tracking stack |
| `lap` or `cython_bbox` (auto via boxmot) | Association speedups |
| `pyyaml` | Config parsing (tracker configs) |

Sample install (CPU only):
```bash
pip install ultralytics boxmot opencv-python pandas
```

Install PyTorch (pick correct command from https://pytorch.org):
```bash
# Example CUDA 12.x wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Optional Performance Tweaks
Set environment for reduced RTSP latency (already in code):
```
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|reorder_queue_size;0|buffer_size;0|stimeout;5000000"
```

#### Verify GPU
```bash
python LocalPC/CHECK_GPU.py
```

#### Run Tracking
```bash
python LocalPC/PY_DEPLOY_EZVIZ3.py
```
Adjust inside script:
| Variable | Meaning |
|----------|---------|
| `VERIFICATION_CODE` | Camera auth code |
| `IP_ADDRESS` | LAN camera IP |
| `YOLO_WEIGHTS` | Path to `best.pt` |
| `REID_WEIGHTS` | Path to ReID model |
| `conf_thres` | YOLO confidence threshold (0.25–0.5 typical) |

#### Suggested `requirements.txt`
```
ultralytics
torch           # (omit here and install manually for proper CUDA)
torchvision
opencv-python
pandas
numpy
boxmot
scipy
lap
pyyaml
```

---

### 3. Nano J1010/

Edge device script integrating PIR + sound + serial feed from ESP32:
- `PKMCANTIKA.py` – Reads motion & sound via Jetson GPIO + parses temperature/humidity from ESP32 over UART.

#### Hardware
| Item | Notes |
|------|-------|
| Jetson J1010 (Jetson Nano family) | JetPack ≥ 5.x recommended |
| PIR Sensor → Pin D17 | Configure as input |
| Sound sensor (digital trigger) → Pin D27 | Active LOW example toggling |
| UART wiring | ESP32 TX (GPIO17) → Jetson RX (ttyTHS1), ESP32 RX (GPIO16) ← Jetson TX (if needed), common GND |
| Logic level | Both are 3.3V tolerant |

#### Enable UART on Jetson
- Disable conflicting serial console if necessary.
- Confirm device: `/dev/ttyTHS1` exists.
- Add your user to dialout / tty groups:
```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

#### System Packages
```bash
sudo apt update
sudo apt install -y python3-pip python3-dev
```

#### Python Dependencies
| Library | Purpose |
|---------|---------|
| `pyserial` | Serial communication |
| `adafruit-blinka` | Board abstraction layer |
| `digitalio` (via Blinka) | GPIO access |
| `RPi.GPIO` (not used) | Not required on Jetson with Blinka |
| `time` / `board` | Standard / Blinka |

Install:
```bash
python3 -m pip install --upgrade pip
pip install pyserial adafruit-blinka
```

Configure Blinka for Jetson (usually auto-detected). Test:
```bash
python3 - <<'EOF'
import board, digitalio
print("Board OK:", board.I2C())
EOF
```

#### Run Script
```bash
python3 "Nano J1010/PKMCANTIKA.py"
```

Expected serial line format from ESP32:
```
TEMP:25.3,HUM:60.2,MOTION:1,SOUND:0
```
(Adapt your ESP32 sketch if you want MOTION/SOUND appended—current Jetson script only expects TEMP/HUM but prints unexpected lines if format differs.)

---

### 4. Inter-Component Data Flow

| Source | Transport | Target | Format |
|--------|-----------|--------|--------|
| ESP32 Sensors (DHT22 / PIR / Mic) | UART (115200, Serial2 pins 16/17) | Jetson J1010 | Plain text lines (`TEMP:..,HUM:..,...`) |
| ESP32 (optional) | MQTT over TLS | Cloud / Broker | JSON (`{"temp":..,"hum":..,"motion":0,"sound":1}`) |
| Jetson Aggregation | Console logs | Developer | Human-readable |
| LocalPC YOLO Tracker | CSV / MP4 | Disk | `goat_tracks_live.csv`, annotated video |
| YOLO Tracker (future) | MQTT / WebSocket (optional) | Dashboard | JSON events |

---
---

Maintainer: (Kry the one behind the lines)
