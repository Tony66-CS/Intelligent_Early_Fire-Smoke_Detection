# 🔥 FireWatch — Real-Time Fire & Smoke Detection

> **YOLO26m transfer learning** + **Groq Llama-3 AI alerts** for early fire detection on live video streams.

---

## Overview

FireWatch is a computer vision system that detects fire and smoke in real time using a fine-tuned YOLO26m model. When a threat is detected, it calls a Groq-hosted Llama-3 LLM to generate a structured emergency alert — danger level, immediate action, and contextual detail — overlaid directly on the video feed.

---

## Features

- **YOLO26m transfer learning** — fine-tuned on fire/smoke datasets with targeted augmentation for varied lighting and smoke density conditions
- **Small-object detection improvements** — catches early-stage fires at distance
- **Real-time inference** — runs on live webcam streams at full resolution (1280×720)
- **Groq Llama-3 AI alerts** — natural-language emergency guidance generated in a background thread so video stays smooth
- **Adaptive danger levels** — `NONE → LOW → MEDIUM → HIGH → CRITICAL` based on consecutive detection frames
- **Visual HUD** — live danger badge, per-class frame counters, FPS display, and AI message overlay
- **Critical flash overlay** — pulsing red border when danger reaches CRITICAL
- **Screenshot capture** — save annotated frames on demand

---

## Requirements

```
Python 3.9+
ultralytics
groq
opencv-python
```

Install all dependencies:

```bash
pip install ultralytics groq opencv-python
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/your-username/firewatch.git
cd firewatch
```

**2. Add your model weights**

Place your trained YOLOv8 weights at:

```
Models/best-2.pt
```

**3. Set your Groq API key**

Get a free key at [console.groq.com](https://console.groq.com), then open `fire_detection_realtime.py` and set:

```python
GROQ_API_KEY = "your_key_here"
```

Or export it as an environment variable and update the client initialization accordingly.

---

## Usage

```bash
python fire_detection_realtime.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot of current frame |
| `R` | Force an immediate Groq AI analysis |
| `M` | Toggle mute (hide AI message overlay) |

---

## Configuration

All tunable parameters are at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `Models/best-2.pt` | Path to YOLO weights |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `CONF_THRESHOLD` | `0.40` | Minimum detection confidence |
| `ALERT_FRAMES` | `3` | Consecutive frames before triggering alert |
| `LLM_COOLDOWN_SEC` | `8` | Minimum seconds between Groq calls |
| `DANGER_WINDOW` | `30` | Frame history window for danger estimation |

**Alternative Groq models:**
- `llama3-70b-8192` — higher accuracy, slightly slower
- `llama-3.1-8b-instant` — fastest response time

---

## Alert Format

Each Groq response follows a structured three-line format:

```
DANGER: [LOW / MEDIUM / HIGH / CRITICAL]
ACTION: [Immediate instruction for people on-site]
DETAIL: [Additional context or warning]
```

---

## Danger Level Logic

Danger is estimated locally (before Groq responds) based on consecutive detection frames:

| Level | Trigger condition |
|-------|------------------|
| `NONE` | No detections |
| `LOW` | Any fire/smoke detected |
| `MEDIUM` | ≥ 3 consecutive frames |
| `HIGH` | ≥ 5 fire frames or ≥ 8 smoke frames |
| `CRITICAL` | ≥ 10 fire frames or ≥ 15 smoke frames |

---

## Model Training Notes

The YOLO26m base model was fine-tuned with:

- **Targeted data augmentation** — HSV jitter, mosaic, and random erasing to handle varied lighting and smoke density
- **Small-object detection improvements** — adjusted anchor sizes and multi-scale training
- **Classes** — `smoke` (index 0), `fire` (index 1)

---

## Project Structure

```
firewatch/
├── fire_detection_realtime.py   # Main script
├── Models/
│   └── best-2.pt                # Trained YOLO weights (not included)
├── screenshots/                 # Saved frames (auto-created)
└── README.md
```

---

## License

MIT
