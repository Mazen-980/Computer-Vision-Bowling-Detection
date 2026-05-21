# 🎳 Bowling AI Detection System

A real-time bowling pin detection and automated scoring system powered by **YOLOv8** deep learning. The system analyzes video footage of a bowling lane to detect the ball, pins, and ball-return car — then automatically computes the score with no manual input required.

Two deployment targets are supported: a **Python desktop pipeline** for full-video processing on PC, and an **Android mobile app** that runs inference directly on the phone.

---

## Features

- **Real-time object detection** — detects ball, ball-return car, standing pins, and fallen pins using YOLOv8
- **Automated scoring** — computes bowling score frame-by-frame with noise filtering
- **Impact gate** — only counts pins that fall *after* the ball makes contact (pre-existing fallen pins are ignored)
- **Consecutive-frame confirmation** — requires 3 consecutive fallen detections before counting, eliminating single-frame YOLO misclassifications
- **Pin identity tracking** — nearest-neighbor tracker assigns persistent IDs to each pin across frames
- **Annotated video output** — draws bounding boxes, labels, and live score overlay on the output video
- **Dual platform** — Python/OpenCV for desktop, TensorFlow Lite for Android

---

## Architecture

```
Video Input
    │
    ▼
Frame Extraction (OpenCV)
    │
    ▼
YOLOv8 Inference
    │
    ├──► Ball Detection
    ├──► Ball-Return Car Detection
    ├──► Standing Pin Detection
    └──► Fallen Pin Detection
              │
              ▼
        Impact Gate ──► (impact_started flag, one-way)
              │
              ▼
        Pin Tracker ──► Nearest-neighbor matching (< 120px radius)
              │
              ▼
        FALLEN_CONFIRM filter ──► 3 consecutive frames required
              │
              ▼
        Score Computation
              │
              ▼
        Annotated Output Video
```

### YOLOv8 Model

The model is built on three stages:

| Stage | Component | Role |
|---|---|---|
| Backbone | CSPDarknet | Multi-scale feature extraction |
| Neck | PANet | Scale-aware feature fusion |
| Head | Detection Layer | Predicts bounding boxes + class scores |

Output tensor shape: `[1, 8, 3549]` — 3549 anchor positions, each with 4 coordinates + 4 class scores.

**Classes detected:**
- `0` — Ball
- `1` — Ball-return car
- `2` — Fallen pin
- `3` — Standing pin

---

## Project Structure

```
bowling-ai/
│
├── main.py                  # Python desktop pipeline
├── model/
│   ├── best.pt              # YOLOv8 weights (PyTorch)
│   └── best.tflite          # TFLite weights (Android)
│
├── BWapplication/           # Android app (Kotlin)
│   └── app/src/main/
│       ├── java/com/example/bwapplication/
│       │   ├── ui/home/     # Main detection fragment
│       │   ├── ui/dashboard/
│       │   └── ui/notifications/
│       └── assets/
│           └── best.tflite
│
└── README.md
```

---

## Getting Started

### Python Desktop Pipeline

**Requirements:**
- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)

**Install dependencies:**
```bash
pip install ultralytics opencv-python
```

**Run on a video:**
```bash
python main.py --input your_video.mp4 --output result.mp4
```

The output video will have bounding boxes, class labels, and a live score overlay drawn on each frame.

---

### Android App

**Requirements:**
- Android Studio
- Android device or emulator (API 26+)
- The `best.tflite` model file placed in `app/src/main/assets/`

**Build & run:**
1. Open the `BWapplication` folder in Android Studio
2. Place `best.tflite` in `app/src/main/assets/`
3. Connect your device and click **Run**

The app uses the phone camera or a selected video to run inference on-device via TensorFlow Lite.

---

## Key Implementation Details

### `impact_started` flag
A one-way gate that becomes `True` once the ball is detected close to the pins. It never resets to `False` within a single roll. This prevents any pins already on the floor before the ball arrives from being counted in the score.

### `FALLEN_CONFIRM = 3`
YOLO can occasionally misclassify a standing pin as fallen for a single frame. By requiring 3 *consecutive* fallen detections before incrementing the score, the system is robust against these transient errors.

### Pin tracker
A lightweight nearest-neighbor tracker using Euclidean distance. If a detected pin in the current frame is within 120px of a pin from the previous frame, they are considered the same physical pin. This avoids the complexity of a full Kalman filter while being sufficient for the slow-moving pins.

### `boxes_close()`
A proximity check (margin = 20px) used to determine when the ball is near the pins. Triggers the `impact_started` gate. Uses coordinate comparison rather than full IoU for speed.

---

## Limitations

- Pin tracker uses fixed 120px radius — can fail if two adjacent pins move simultaneously
- `FALLEN_CONFIRM = 3` may miss pins knocked down in fewer than 3 frames at the sampled rate
- All annotated frames are held in RAM (memory scales with video length)
- Model trained on a specific lane; accuracy may drop under different lighting or camera angles
- No temporal smoothing on bounding boxes (can cause minor jitter)

---

## Future Work

- Replace nearest-neighbor tracker with **ByteTrack** or **Kalman filter** for more robust ID management
- Export model as **INT8 quantized** for 4× speedup and 75% size reduction on mobile
- Stream annotated frames to disk instead of RAM to remove the memory limitation
- Add **SQLite match history** on Android to persist and compare scores across sessions
- Train on additional camera angles and lighting conditions for better generalization

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Desktop Pipeline | Python, OpenCV |
| Mobile App | Android (Kotlin) |
| Mobile Inference | TensorFlow Lite |
| Model Architecture | CSPDarknet + PANet |

---

## License

MIT License — feel free to use, modify, and distribute.
