# AI-Based Driver Drowsiness and Distraction Detection System

> **MCA Final Year Project — 2024**  
> Real-time driver monitoring using Computer Vision and Deep Learning

---

## Overview

DrowsyGuard is a real-time driver safety system that detects:

- **Eye closure** — Eye Aspect Ratio (EAR) via MediaPipe FaceMesh
- **Yawning** — Mouth Aspect Ratio (MAR)
- **Head pose** — Pitch/Yaw/Roll via OpenCV solvePnP
- **Phone usage** — YOLOv8 object detection
- **Distraction** — Fused multi-signal scoring

Alerts are delivered via audio (pygame) and text-to-speech (pyttsx3). All events are logged to SQLite and visualised on a Flask + Socket.IO web dashboard.

---

## Quick Start

### 1. Clone / extract the project
```bash
cd Driver-Drowsiness-System
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `dlib` requires CMake. On Windows run `pip install cmake` first.  
> For GPU support install the CUDA-enabled PyTorch wheel from https://pytorch.org

### 4. Initialise the database
```bash
python database/init_db.py
# Add sample data for testing the UI:
python database/init_db.py --sample
```

### 5. Run the application
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Project Structure

```
Driver-Drowsiness-System/
├── app.py                      # Flask application entry point
├── requirements.txt
├── README.md
│
├── modules/                    # Detection modules
│   ├── face_detector.py        # MediaPipe FaceMesh wrapper
│   ├── eye_detector.py         # EAR-based eye closure detection
│   ├── yawn_detector.py        # MAR-based yawn detection
│   ├── head_pose.py            # solvePnP head pose estimation
│   ├── phone_detector.py       # YOLOv8 phone detection
│   ├── distraction_detector.py # Multi-signal fusion scorer
│   └── alert_system.py         # Audio + TTS alert manager
│
├── database/
│   ├── models.py               # SQLAlchemy ORM models
│   ├── db_manager.py           # High-level data access layer
│   ├── schema.sql              # Raw SQL schema + views
│   └── init_db.py              # Standalone DB initialiser
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── logs.html
│   ├── reports.html
│   └── settings.html
│
├── static/
│   ├── css/style.css
│   └── js/                     # dashboard.js, logs.js, reports.js …
│
├── training/
│   ├── train_drowsiness_cnn.py # PyTorch CNN for eye state
│   ├── train_yolo_phone.py     # YOLOv8 fine-tuning
│   ├── evaluate_real.py        # REAL metrics on labelled images (EAR/MAR/sunglasses)
│   ├── extract_frames.py       # video -> labelled frames for your own data
│   └── dataset_info.md         # Dataset download links
│
├── tests/
│   ├── test_detectors.py
│   ├── test_flask_app.py
│   ├── test_alert_system.py
│   ├── test_database.py
│   └── conftest.py
│
├── yolov8_model/               # YOLOv8 weights (yolov8n.pt)
├── alerts/                     # Alert audio files (.wav/.mp3)
├── screenshots/                # Auto-captured alert screenshots
├── dataset/                    # Training datasets
├── reports/                    # Generated evaluation reports
└── documentation/              # Research paper, project report, PPT
```

---

## Configuration

Thresholds can be adjusted at runtime via the **Settings** page or by editing `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | 0.25 | EAR below this → eyes closed |
| `EAR_CONSEC_FRAMES` | 20 | Frames before drowsiness alert |
| `MAR_THRESHOLD` | 0.60 | MAR above this → yawning |
| `MAR_CONSEC_FRAMES` | 15 | Frames before yawn alert |
| `HEAD_PITCH_THRESHOLD` | 20° | Vertical head tilt limit |
| `HEAD_YAW_THRESHOLD` | 30° | Horizontal head turn limit |
| `PHONE_CONF_THRESHOLD` | 0.50 | YOLOv8 minimum confidence |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/ -v -m unit

# With coverage report
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

---

## Model Training

### Eye State CNN
```bash
# Download dataset first (see training/dataset_info.md)
python training/train_drowsiness_cnn.py
```

### YOLOv8 Phone Detector
```bash
python training/train_yolo_phone.py
```

### Evaluate All Models
```bash
python training/evaluate_real.py
# Reports saved to reports/
```

---

## Detection Accuracy

**No accuracy figures are claimed until you measure them.** Run the evaluation scripts
(see TRAINING_GUIDE.md) and paste the real numbers from `reports/` here.
Always state which method each result belongs to: pretrained model, custom-trained model, or heuristic.

| Module | Method type | Test images | Accuracy | Precision | Recall | F1 |
|--------|-------------|-------------|----------|-----------|--------|----|
| Eye state CNN | custom-trained | (fill) | (fill) | (fill) | (fill) | (fill) |
| Eye closure (EAR) | heuristic | (fill) | (fill) | (fill) | (fill) | (fill) |
| Yawn (MAR) | heuristic | (fill) | (fill) | (fill) | (fill) | (fill) |
| Sunglasses | heuristic | (fill) | (fill) | (fill) | (fill) | (fill) |
| Phone (YOLOv8n COCO) | pretrained | (fill) | mAP50 (fill) | | | |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.0, Flask-SocketIO |
| Computer Vision | OpenCV 4.10, MediaPipe 0.10 |
| Deep Learning | PyTorch 2.3, YOLOv8 (Ultralytics) |
| Database | SQLite 3, SQLAlchemy 2.0 |
| Frontend | HTML5, CSS3, JavaScript, Chart.js, Socket.IO |
| Audio | pygame, pyttsx3 |
| Testing | pytest, pytest-flask |

---

## Dataset Links

| Dataset | URL |
|---------|-----|
| MRL Eye Dataset | https://mrl.cs.vsb.cz/eyedataset |
| Drowsiness Kaggle | https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset |
| Roboflow Phone | https://universe.roboflow.com/roboflow-100/cell-phones-detection |
| NTHU Drowsy Driver | http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |

---

## License

This project is developed for academic purposes as an MCA Final Year Project.

---

## Author

**MCA Final Year Student — 2024**  
Department of Computer Applications  
*AI-Based Driver Drowsiness and Distraction Detection System*
