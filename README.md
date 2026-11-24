# AI-Based Driver Drowsiness and Distraction Detection System

> **MCA Final Year Project — 2026**

Real-time driver monitoring using Computer Vision and Deep Learning.

---

## Overview

**DrowsyGuard** is a real-time driver safety and monitoring system designed to detect driver drowsiness and distraction using computer vision and deep learning techniques.

The system detects the following driver behaviours:

- **Eye Closure Detection** – Uses Eye Aspect Ratio (EAR) through MediaPipe FaceMesh to identify prolonged eye closure.
- **Yawning Detection** – Uses Mouth Aspect Ratio (MAR) to identify signs of drowsiness.
- **Head Pose Estimation** – Estimates Pitch, Yaw and Roll angles using OpenCV solvePnP.
- **Phone Usage Detection** – Uses YOLOv8 object detection to identify mobile phone usage.
- **Distraction Detection** – Combines multiple detection signals to identify distracted driving.

Alerts are delivered through audio notifications using pygame and text-to-speech using pyttsx3.

All detection events are stored in an SQLite database and displayed through a Flask and Socket.IO web dashboard.

## Key Features

- Real-time driver face monitoring.
- Eye closure and drowsiness detection.
- Yawning detection.
- Head pose estimation.
- Mobile phone usage detection.
- Sunglasses detection.
- Multi-signal distraction detection.
- Audio and voice-based alerts.
- SQLite database for event logging.
- Interactive web dashboard.
- Detection history and reports.
- Model training and evaluation scripts.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Sowbhs-code/AI-Driver-Drowsiness-Detection.git
cd AI-Driver-Drowsiness-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:**

- Install CMake if required by the dlib dependency.
- For GPU support, install the appropriate CUDA-enabled PyTorch version from https://pytorch.org.
- Ensure that all required Python dependencies are installed before running the application.

### 4. Initialise the Database

```bash
python database/init_db.py
```

To add sample data for testing the dashboard:

```bash
python database/init_db.py --sample
```

### 5. Run the Application

```bash
python app.py
```

Open the following URL in your browser:

http://127.0.0.1:5000

---

## Project Structure

```text
AI-Driver-Drowsiness-Detection/
│
├── app.py
├── README.md
├── TRAINING_GUIDE.md
├── requirements.txt
├── req_app.txt
│
├── alerts/
│   ├── alert_distracted.wav
│   ├── alert_drowsy.wav
│   ├── alert_phone.wav
│   └── alert_yawn.wav
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py
│   ├── init_db.py
│   ├── models.py
│   └── schema.sql
│
├── documentation/
│   ├── ppt_content.md
│   ├── project_report.md
│   └── research_paper.md
│
├── models/
│   ├── eye_cnn.keras
│   ├── eye_cnn_v2.keras
│   ├── sunglasses_cnn.keras
│   ├── yawn_cnn.keras
│   └── yawn_cnn_v2.keras
│
├── modules/
│   ├── __init__.py
│   ├── alert_system.py
│   ├── cnn_models.py
│   ├── distraction_detector.py
│   ├── eye_detector.py
│   ├── face_detector.py
│   ├── head_pose.py
│   ├── phone_detector.py
│   ├── sunglasses_detector.py
│   └── yawn_detector.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── dashboard.js
│       ├── logs.js
│       ├── main.js
│       ├── reports.js
│       └── settings.js
│
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── index.html
│   ├── login.html
│   ├── logs.html
│   ├── register.html
│   ├── reports.html
│   └── settings.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_alert_system.py
│   ├── test_database.py
│   ├── test_detectors.py
│   └── test_flask_app.py
│
├── training/
│   ├── dataset_info.md
│   ├── evaluate_real.py
│   ├── extract_frames.py
│   ├── train_drowsiness_cnn.py
│   └── train_yolo_phone.py
│
├── yolov8_model/
│   └── yolov8n.pt
│
├── train_eye.py
├── train_sunglasses.py
├── train_yawn.py
└── train_yawn_v2.py
```

---

## Configuration

Detection thresholds can be adjusted through the Settings page or by editing the application configuration in `app.py`.

| Parameter | Default | Description |
|---|---:|---|
| EAR_THRESHOLD | 0.25 | EAR below this value indicates closed eyes |
| EAR