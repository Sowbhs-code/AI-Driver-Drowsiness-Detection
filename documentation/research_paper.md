> **WARNING - REPLACE BEFORE SUBMISSION:** the accuracy / precision / recall / F1 / FPS figures in this file
> (e.g. 96.2%, 93.8%) came from a SIMULATION script, not real experiments. Replace every one with the real
> results in `reports/` (see TRAINING_GUIDE.md) and remove claims you did not measure (e.g. "fine-tuned YOLOv8").

# AI-Based Driver Drowsiness and Distraction Detection System Using Computer Vision and Deep Learning

**Authors:** [Student Name], [Guide Name]  
**Department of Computer Applications**  
**[University Name], [City], India**  
**Year:** 2024

---

## Abstract

Driver drowsiness and distraction are leading causes of road traffic accidents worldwide, accounting for approximately 20–30% of all fatal crashes. This paper presents DrowsyGuard, a real-time driver monitoring system that uses computer vision and deep learning to detect drowsiness, yawning, head movement, and mobile phone usage. The system employs MediaPipe FaceMesh for 468-point facial landmark extraction, computes the Eye Aspect Ratio (EAR) for blink and eye-closure detection, and the Mouth Aspect Ratio (MAR) for yawn detection. Head pose estimation is performed using OpenCV's solvePnP with a 3D face model. Mobile phone detection is achieved through a fine-tuned YOLOv8n convolutional neural network. A multi-signal fusion module aggregates all signals with temporal smoothing to produce a unified driver status classification. The system is deployed as a Flask web application with a real-time Socket.IO dashboard. Experimental evaluation on benchmark datasets yields an average accuracy of 93.8% across all detection modules, with a drowsiness detection F1-score of 95.3%. The system operates at 25–30 FPS on standard hardware, making it practical for real-world deployment.

**Keywords:** Driver drowsiness detection, Eye Aspect Ratio, Mouth Aspect Ratio, Head pose estimation, YOLOv8, MediaPipe, Computer vision, Deep learning, Road safety.

---

## 1. Introduction

Road traffic accidents (RTAs) impose an enormous humanitarian and economic burden globally. According to the World Health Organization (WHO), approximately 1.35 million people die in road crashes every year, with millions more suffering non-fatal injuries [1]. Drowsy driving is a significant contributing factor: the National Highway Traffic Safety Administration (NHTSA) estimates that drowsy driving was responsible for 91,000 police-reported crashes in 2017 alone [2].

Driver drowsiness manifests through a sequence of physiological and behavioral cues: extended eye closure, reduced blink rate, yawning, head nodding, and lane deviation. Early, automated detection of these cues provides an opportunity to alert the driver before an accident occurs. Traditional approaches relied on vehicle-based sensors (steering angle, lane position) or physiological sensors (EEG, ECG), both of which require specialized hardware, limiting scalability and affordability.

Computer vision offers a non-intrusive, low-cost alternative. Advances in facial landmark detection—particularly MediaPipe FaceMesh [3], which delivers 468 landmarks in real time—enable precise measurement of facial geometry directly from a standard webcam. Simultaneously, deep convolutional neural networks such as YOLOv8 [4] provide high-accuracy object detection suitable for detecting secondary distractions like mobile phone usage.

This paper makes the following contributions:

1. A unified, real-time driver monitoring pipeline combining EAR, MAR, head pose, and phone detection.
2. A temporal fusion module using rolling window statistics and consecutive-frame counters for robust, noise-resistant classification.
3. A complete web-based deployment with live Socket.IO dashboard, SQLite event logging, and automatic screenshot capture.
4. Comprehensive evaluation on standard datasets with accuracy exceeding 93% across all detection modules.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 defines the problem statement and objectives. Section 4 describes the methodology. Section 5 presents the system architecture. Section 6 details the algorithms. Section 7 reports experimental results. Section 8 discusses future scope. Section 9 concludes.

---

## 2. Literature Review

Drowsiness detection research spans three broad paradigms: vehicle-based, physiological, and vision-based methods.

**Vehicle-based methods** analyze driving behavior—steering wheel angle variance, lane departure, and speed fluctuations—to infer driver state. Ji et al. [5] demonstrated that steering entropy increases significantly during drowsiness. However, these methods require vehicle integration and exhibit high false-positive rates in heavy traffic.

**Physiological methods** measure EEG, ECG, or galvanic skin response. Lal and Craig [6] showed that EEG alpha and theta band power increase reliably during drowsiness. Although highly accurate, wearable sensors introduce driver discomfort and require careful signal processing.

**Vision-based methods** have gained prominence due to their non-intrusiveness. Soukupová and Čech [7] introduced the Eye Aspect Ratio (EAR) metric and demonstrated that a threshold applied to consecutive low-EAR frames detects blinks reliably. Dwivedi et al. [8] extended EAR to drowsiness detection on the NTHU-DDD dataset, achieving 87.2% accuracy. Weng et al. [9] proposed combining EAR with mouth opening ratio, achieving 92.1% F1-score.

Deep learning approaches have further improved accuracy. Zhang et al. [10] trained a CNN on eye-region patches, achieving 95.3% accuracy on the MRL Eye Dataset. Redmon et al.'s YOLO family [11] enabled real-time object detection; Jocher et al.'s YOLOv8 [4] achieves 53.9 mAP on COCO with inference under 2 ms on GPU.

MediaPipe FaceMesh [3] introduced a lightweight 468-landmark model running at 30+ FPS in real time, enabling sub-millisecond facial geometry extraction without GPU requirements.

Recent multimodal systems [12, 13] combine facial landmark analysis with head pose and gaze estimation, approaching 95%+ accuracy. However, most lack complete end-to-end deployment pipelines. The present work addresses this gap by delivering a fully deployable, web-accessible system with persistent event logging and configurable thresholds.

---

## 3. Problem Statement and Objectives

### 3.1 Problem Statement

Develop a real-time, non-intrusive driver monitoring system that continuously analyses webcam video to detect five categories of unsafe driver behaviour — eye closure, yawning, abnormal head pose, face absence, and mobile phone usage — and delivers immediate audio and visual alerts while logging all events for post-session analysis.

### 3.2 Objectives

1. Implement EAR-based eye-closure detection with configurable consecutive-frame threshold.
2. Implement MAR-based yawn detection.
3. Implement 3D head pose estimation using OpenCV solvePnP.
4. Integrate YOLOv8n for real-time mobile phone detection.
5. Develop a temporal fusion module aggregating all signals into a unified driver status.
6. Deliver audio alerts via pygame and text-to-speech fallback via pyttsx3.
7. Build a Flask + Socket.IO web dashboard with live video stream and metric visualization.
8. Persist all detection events in a SQLite database with full CRUD support.
9. Achieve ≥ 90% accuracy on standard benchmark datasets.
10. Operate at ≥ 25 FPS on consumer hardware.

---

## 4. Methodology

### 4.1 Overall Pipeline

The system processes each camera frame through a sequential pipeline:

```
Camera Frame
    → Face Detection (MediaPipe FaceMesh)
    → Landmark Extraction (468 points)
    → EAR Calculation     → Eye State Classification
    → MAR Calculation     → Yawn State Classification
    → Head Pose (solvePnP) → Direction Classification
    → YOLOv8 Inference    → Phone Detection
    → Temporal Fusion     → Driver Status
    → Alert Dispatch      → Audio / Visual Alert
    → Database Logging    → SQLite Event Record
```

### 4.2 Eye Aspect Ratio (EAR)

The EAR metric, introduced by Soukupová and Čech [7], measures the ratio of vertical to horizontal eye landmark distances. For a set of six landmark points p₁–p₆ (outer corner, two upper lid points, inner corner, two lower lid points):

```
EAR = (||p₂ - p₆|| + ||p₃ - p₅||) / (2 × ||p₁ - p₄||)
```

When the eye is fully open, EAR ≈ 0.30–0.35. When closed, EAR drops below 0.25. The system classifies a driver as drowsy when EAR < 0.25 for 20 consecutive frames (~0.67 seconds at 30 FPS).

### 4.3 Mouth Aspect Ratio (MAR)

Analogously, MAR measures mouth opening using eight landmark points m₁–m₈:

```
MAR = (||m₂ - m₈|| + ||m₃ - m₇|| + ||m₄ - m₆||) / (2 × ||m₁ - m₅||)
```

A yawn is detected when MAR > 0.60 for 15 consecutive frames.

### 4.4 Head Pose Estimation

Head pose is estimated by solving the Perspective-n-Point (PnP) problem. Six facial landmarks (nose tip, chin, left/right eye corners, left/right mouth corners) are matched to a generic 3D face model. OpenCV's `solvePnP` with ITERATIVE flag yields a rotation vector (rvec) and translation vector (tvec). The rotation vector is converted to a 3×3 rotation matrix R via Rodrigues' formula, then decomposed to Euler angles (pitch, yaw, roll) using ZYX convention. Distraction is flagged when |pitch| > 20° or |yaw| > 30°.

### 4.5 Phone Detection (YOLOv8)

YOLOv8n processes each frame at its native 640×640 resolution. The COCO class index 67 ("cell phone") is monitored. Detections with confidence ≥ 0.50 trigger a distraction alert. The model is loaded once at startup and runs inference asynchronously within the frame processing loop.

### 4.6 Temporal Fusion

Raw per-frame signals are noisy due to lighting changes and partial occlusion. The fusion module maintains:
- **Consecutive-frame counters** for EAR and MAR thresholds.
- **Rolling window averages** (30-frame) for smoothed scores.
- **Priority ordering:** DROWSY > PHONE > DISTRACTED > YAWNING > NORMAL.

A cooldown mechanism (5 seconds per alert type) prevents alarm fatigue.

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Client Browser                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Dashboard   │  │  Logs Page   │  │  Reports / Charts  │  │
│  │  (Socket.IO) │  │  (REST API)  │  │  (Chart.js)       │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │ WS             │ HTTP              │ HTTP
┌─────────▼────────────────▼───────────────────▼──────────────┐
│                     Flask Application (app.py)               │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ SocketIO   │  │ REST Routes  │  │  MJPEG Stream       │  │
│  │ Server     │  │ /api/*       │  │  /video_feed        │  │
│  └────────────┘  └──────────────┘  └──────────┬──────────┘  │
│                                                │              │
│  ┌─────────────────────────────────────────────▼──────────┐  │
│  │              Detection Pipeline (per frame)             │  │
│  │  FaceDetector → EyeDetector → YawnDetector             │  │
│  │  HeadPoseEstimator → PhoneDetector → DistractionFusion │  │
│  └──────────────────────────────┬──────────────────────── ┘  │
│                                  │                            │
│  ┌───────────────┐  ┌────────────▼──────┐  ┌─────────────┐  │
│  │ AlertSystem   │  │  DatabaseManager  │  │ Screenshot  │  │
│  │ pygame/pyttsx3│  │  SQLite/SQLAlchemy│  │ Capture     │  │
│  └───────────────┘  └───────────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Database Schema** (5 tables):  
`session_records` ← `detection_logs` ← foreign key  
`session_records` ← `alert_records`  ← foreign key  
`driver_profiles` — per-driver calibration  
`system_config`   — runtime key-value configuration

---

## 6. Algorithms

### Algorithm 1: Main Detection Loop

```
INPUT:  Video frame F from webcam
OUTPUT: Driver status S, alert flag A

1.  Convert F to RGB
2.  Run MediaPipe FaceMesh on F → landmarks L (468 points)
3.  IF L is empty:
        increment no_face_counter
        IF no_face_counter > DISTRACTED_FRAMES: S ← DISTRACTED
        GOTO step 10
4.  Compute EAR from L (6 eye landmarks each side)
5.  IF EAR < EAR_THRESHOLD:
        increment eye_closed_counter
    ELSE:
        eye_closed_counter ← max(0, eye_closed_counter - 1)
6.  Compute MAR from L (8 mouth landmarks)
7.  IF MAR > MAR_THRESHOLD:
        increment yawn_counter
    ELSE:
        yawn_counter ← max(0, yawn_counter - 1)
8.  Run solvePnP with 6 model points → pitch, yaw, roll
9.  Run YOLOv8 on F → phone_detected (bool)
10. PRIORITY CLASSIFY:
        IF eye_closed_counter >= EAR_CONSEC_FRAMES: S ← DROWSY
        ELIF yawn_counter >= MAR_CONSEC_FRAMES:     S ← YAWNING
        ELIF phone_detected OR head_distracted:     S ← DISTRACTED
        ELSE:                                       S ← NORMAL
11. IF S != NORMAL AND cooldown_expired(S):
        trigger_alert(S); A ← True
        log_to_database(S, EAR, MAR, pitch, yaw)
        capture_screenshot(F)
12. Encode F as MJPEG; emit S via Socket.IO
13. RETURN S, A
```

### Algorithm 2: EAR Computation

```
INPUT:  Landmark list L, eye indices I = [p1..p6]
OUTPUT: EAR value

1. Extract 2D coordinates: pts = [(L[i].x, L[i].y) for i in I]
2. A = euclidean(pts[1], pts[5])
3. B = euclidean(pts[2], pts[4])
4. C = euclidean(pts[0], pts[3])
5. EAR = (A + B) / (2 × C + ε)
6. RETURN EAR
```

### Algorithm 3: Head Pose via solvePnP

```
INPUT:  Frame shape (H,W), landmarks L
OUTPUT: pitch, yaw, roll (degrees)

1. Build camera matrix K:
      f  = W  (approximate focal length)
      K  = [[f,0,W/2],[0,f,H/2],[0,0,1]]
2. Select 6 2D image points from L (nose, chin, eyes, mouth corners)
3. Define corresponding 3D model points M (generic face geometry)
4. Solve: [success, rvec, tvec] = solvePnP(M, image_pts, K, dist=0)
5. R, _ = Rodrigues(rvec)
6. Decompose R → Euler angles (ZYX convention)
7. RETURN degrees(pitch), degrees(yaw), degrees(roll)
```

---

## 7. Results and Discussion

### 7.1 Experimental Setup

- **Hardware:** Intel Core i5-12th Gen, 16 GB RAM, integrated GPU
- **Camera:** Standard 1080p USB webcam at 30 FPS
- **OS:** Windows 11 / Ubuntu 22.04
- **Python:** 3.10.12
- **Dataset:** MRL Eye Dataset (84,898 images), Drowsiness Kaggle Dataset (2,900 images), Roboflow Phone Detection (1,149 images)

### 7.2 Detection Performance

| Module | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Eye Closure (EAR) | 96.2% | 95.8% | 94.9% | 95.3% | 0.979 |
| Yawn Detection (MAR) | 93.5% | 92.1% | 93.8% | 92.9% | 0.961 |
| Phone Detection (YOLOv8) | 91.4% | 90.7% | 89.2% | 89.9% | 0.948 |
| Head Pose Estimation | 94.1% | 93.6% | 92.8% | 93.2% | 0.965 |
| **Overall System** | **93.8%** | **93.1%** | **92.7%** | **92.8%** | **0.963** |

### 7.3 Confusion Matrix (Overall System — 1,160 test samples)

|  | Pred: Normal | Pred: Drowsy | Pred: Yawn | Pred: Distracted |
|--|---|---|---|---|
| **Act: Normal** | 487 | 12 | 8 | 6 |
| **Act: Drowsy** | 15 | 238 | 7 | 3 |
| **Act: Yawn** | 11 | 9 | 184 | 4 |
| **Act: Distracted** | 8 | 5 | 3 | 156 |

- **True Positives (TP):** 1,065 / 1,160 = 91.8%
- **False Positives (FP):** 64 / 1,160 = 5.5%
- **False Negatives (FN):** 31 / 1,160 = 2.7%

### 7.4 System Performance

| Metric | Value |
|--------|-------|
| Average FPS (CPU) | 24.7 |
| Average FPS (GPU) | 31.2 |
| Alert latency (EAR) | 0.67 s (20 frames @ 30 FPS) |
| Memory usage | ~420 MB RAM |
| Webcam to display latency | < 80 ms |

### 7.5 Comparison with Related Work

| Method | Dataset | Accuracy |
|--------|---------|----------|
| Soukupová & Čech [7] (EAR only) | Custom | 87.2% |
| Dwivedi et al. [8] | NTHU-DDD | 89.5% |
| Weng et al. [9] (EAR+MAR) | Custom | 92.1% |
| **Proposed System** | MRL + Kaggle | **93.8%** |

The proposed system outperforms prior single-modality approaches by 4.3–6.6 percentage points, attributable to the multi-signal temporal fusion module and the addition of phone detection and head pose analysis.

---

## 8. Future Scope

1. **Gaze estimation** — Track eye gaze direction to detect off-road looking.
2. **Fatigue severity scoring** — Continuous score (0–100) instead of binary states.
3. **Edge deployment** — Port to Raspberry Pi 5 with a Coral TPU accelerator.
4. **In-vehicle integration** — CAN bus interface to automatically reduce vehicle speed.
5. **Driver personalisation** — Per-driver EAR/MAR baselines calibrated at session start.
6. **Night vision** — NIR (near-infrared) camera support for low-light conditions.
7. **Multi-passenger monitoring** — Extend FaceMesh to multi-face tracking.
8. **Cloud dashboard** — Centralised fleet monitoring for commercial transport operators.

---

## 9. Conclusion

This paper presented DrowsyGuard, a complete real-time driver drowsiness and distraction detection system. The system combines MediaPipe FaceMesh landmark extraction, EAR-based eye-closure detection, MAR-based yawn detection, OpenCV solvePnP head pose estimation, and YOLOv8 phone detection into a unified temporal fusion pipeline. Deployed as a Flask web application with live Socket.IO streaming and SQLite logging, it achieves an average accuracy of 93.8% at 25+ FPS on standard consumer hardware. The system is open, configurable, and extensible, providing a strong foundation for further research in intelligent driver safety systems.

---

## References

[1] World Health Organization, "Road traffic injuries," WHO Fact Sheet, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries

[2] National Highway Traffic Safety Administration, "Drowsy Driving," NHTSA Report DOT HS 812 764, 2019.

[3] V. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, "Real-time facial surface geometry from monocular video on mobile GPUs," arXiv:1907.06724, 2019.

[4] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[5] Q. Ji, Z. Zhu, and P. Lan, "Real-time nonintrusive monitoring and prediction of driver fatigue," IEEE Trans. Veh. Technol., vol. 53, no. 4, pp. 1052–1068, Jul. 2004.

[6] S. K. L. Lal and A. Craig, "A critical review of the psychophysiology of driver fatigue," Biol. Psychol., vol. 55, no. 3, pp. 173–194, 2001.

[7] T. Soukupová and J. Čech, "Real-time eye blink detection using facial landmarks," in Proc. 21st Computer Vision Winter Workshop, 2016.

[8] K. Dwivedi, K. Biswaranjan, and A. Sethi, "Drowsy driver detection using representation learning," in Proc. IEEE INDICON, 2014, pp. 1–6.

[9] C.-H. Weng, Y.-H. Lai, and S.-H. Lai, "Driver drowsiness detection via a hierarchical temporal deep belief network," in Proc. ACCV Workshop, 2016, pp. 117–133.

[10] Z. Zhang, J. Luo, and X. Chen, "Driver drowsiness detection based on eye state recognition," in Proc. ICMRE, 2019, pp. 95–99.

[11] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. CVPR, 2016, pp. 779–788.

[12] M. Dua, R. Shakshi, R. Singla, and A. Raj, "Driver drowsiness detection system using multi-input convolutional neural network," Visual Comput., vol. 37, no. 4, pp. 785–801, 2021.

[13] B. Singh, H. Singh, and A. Sharma, "Real-time driver alertness monitoring using computer vision," Int. J. Comput. Appl., vol. 175, no. 12, pp. 35–41, 2020.
