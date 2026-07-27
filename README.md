[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/J3J11AMS7S)
# SensorOS

A real-time physical security monitoring system that turns an IP camera feed into a live digital twin of a monitored space. It detects and tracks people and vehicles, scores their behaviour for anomalies, and presents everything as an interactive operator dashboard.

![SensorOS dashboard — live 2D ground-plane view with alert history](docs/dashboard.png)

---

## What it does

1. **Perceives** — A YOLOv8 model (TensorRT-accelerated) detects objects in each camera frame. ByteTrack assigns stable IDs across frames. A homography transform converts every detection from pixel coordinates into real-world metres, placing each entity on a calibrated ground plane.

2. **Scores** — An anomaly engine combines rule-based heuristics (dwell time, velocity, zone membership) with a lightweight LSTM autoencoder trained on normal movement patterns. Each entity receives a fused anomaly score every frame; scores above threshold trigger an alert.

3. **Visualises** — A FastAPI server broadcasts live state over WebSocket to two browser-based views:
   - **Dashboard** (`/`) — 2D world-space map with zone overlays, per-entity markers, alert history, and an LLM scene-query panel backed by Claude.
   - **Path Visualizer** (`/path-viz`) — dedicated view for movement history: fading per-entity trail traces, an accumulated position heatmap, and dwell-spot markers with adjustable decay and fade settings.

---

## Architecture

```
Camera (RTSP)
    │
    ▼
┌─────────────┐   scene_state (Redis)   ┌──────────────┐
│  Perception │ ──────────────────────► │    Anomaly   │
│  YOLOv8 +   │                         │  LSTM + rules│
│  ByteTrack  │                         └──────┬───────┘
└─────────────┘                                │ anomaly_scores (Redis)
                                               ▼
                                        ┌──────────────┐
                                        │     Twin     │
                                        │  FastAPI /   │
                                        │  WebSocket   │
                                        └──────┬───────┘
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              Dashboard            Path Visualizer
                              (port 8080)          (/path-viz)
```

Each component runs in its own Docker container. Redis is the message bus and short-term store; there is no persistent database.

---

## Quick start

### Prerequisites

- NVIDIA GPU with CUDA + TensorRT
- Docker with the NVIDIA container runtime
- An RTSP IP camera

### 1. Configure

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|---|---|
| `CAMERA_RTSP_URL` | Full RTSP URL of your camera, e.g. `rtsp://user:pass@192.168.1.x/stream1` |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | Camera resolution |
| `CAMERA_FPS` | Target frame rate |
| `REDIS_PASSWORD` | Password for the internal Redis instance |
| `ANTHROPIC_API_KEY` | Enables the LLM scene-query panel (optional) |

### 2. Calibrate

Calibration maps image pixels to real-world metre coordinates. Run this once with your camera in its final position:

```bash
python scripts/capture_calibration.py   # capture reference image
python calibration/compute_homography.py
```

The resulting matrix is saved to `calibration/calibration.json` and loaded automatically at startup.

### 3. Run

```bash
docker compose up --build
```

Open **`http://localhost:8080`** for the main dashboard, or **`http://localhost:8080/path-viz`** for the movement history view.

---

## Project layout

```
sensoros/
├── src/
│   ├── perception/     # YOLOv8 detection, ByteTrack, homography, zone assignment
│   ├── anomaly/        # Feature extraction, LSTM autoencoder, rule-based scorer
│   ├── twin/           # FastAPI server, WebSocket broadcast, LLM query endpoint
│   │   └── static/
│   │       ├── index.html              # Main operator dashboard
│   │       ├── path_visualizer.html    # Standalone movement history viewer
│   │       └── plugins/
│   │           └── path_visualizer.js  # Trail / heatmap / dwell rendering class
│   └── shared/         # ZeroMQ pub/sub message bus
├── calibration/        # Homography scripts and calibration data
├── configs/
│   └── zone_config.json  # Zone polygons, alert thresholds
├── models/             # YOLOv8 ONNX / TensorRT engine files
├── docker/             # Per-service Dockerfiles
└── docker-compose.yml
```

---

## Tracked object classes

People, cars, motorcycles, trucks, cats, and dogs (COCO classes 0, 2, 3, 7, 15, 16).

## Anomaly scoring

The fused score is a weighted combination of:
- **Rule score** — flags excessive dwell time, very low velocity in a high-alert zone, or presence in restricted areas.
- **LSTM score** — reconstruction error from an autoencoder trained on 12-dimensional feature vectors (position, velocity, dwell time, trajectory linearity, time-of-day encoding, zone alert level).

Alerts fire when the fused score exceeds the threshold defined in `configs/zone_config.json`.
=======