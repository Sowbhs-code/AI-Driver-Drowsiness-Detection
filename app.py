"""
============================================================
AI-Based Driver Drowsiness and Distraction Detection System
Main Flask Application Entry Point
Author: MCA Final Year Project
============================================================
"""

import os
import cv2
import time
import base64
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# Local modules
from modules.eye_detector import EyeDetector
from modules.yawn_detector import YawnDetector
from modules.head_pose import HeadPoseEstimator
from modules.phone_detector import PhoneDetector
from modules.face_detector import FaceDetector
from modules.sunglasses_detector import SunglassesDetector
from modules.cnn_models import CNNModels
from modules.alert_system import AlertSystem
from database.db_manager import DatabaseManager
from database.models import db, DetectionLog, AlertRecord, SessionRecord, User

# ─── App Configuration ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drowsiness_detection_secret_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'drowsiness.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SCREENSHOT_DIR'] = os.path.join(BASE_DIR, 'screenshots')
app.config['ALERT_AUDIO_DIR'] = os.path.join(BASE_DIR, 'alerts')

db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Flask-Login Setup ────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Global State ─────────────────────────────────────────────────────────────
camera = None
camera_lock = threading.Lock()
detection_active = False
current_session_id = None

driver_state = {
    'status': 'NORMAL',          # NORMAL | DROWSY | DISTRACTED | ALERT
    'eye_status': 'Open',
    'yawn_status': 'No Yawn',
    'head_pose': 'Forward',
    'phone_detected': False,
    'ear': 0.0,                  # Eye Aspect Ratio
    'mar': 0.0,                  # Mouth Aspect Ratio
    'head_pitch': 0.0,
    'head_yaw': 0.0,
    'head_roll': 0.0,
    'drowsy_frames': 0,
    'yawn_frames': 0,
    'distracted_frames': 0,
    'alert_count': 0,
    'session_start': None,
    'fps': 0.0,
    'face_detected': False,
    'sunglasses_detected': False,   # driver is wearing dark/mirrored eyewear
    'detection_mode': 'EAR',        # 'EAR' (normal) | 'HEAD_POSE' (glasses on, eyes hidden)
}

# ─── Detection Thresholds ─────────────────────────────────────────────────────
CONFIG = {
    'EAR_THRESHOLD': 0.25,       # Below this = eyes closed
    'EAR_CONSEC_FRAMES': 8,     # Consecutive frames for drowsiness
    'MAR_THRESHOLD': 0.6,        # Above this = yawning
    'MAR_CONSEC_FRAMES': 8,
    'HEAD_PITCH_THRESHOLD': 20,  # Degrees
    'HEAD_YAW_THRESHOLD': 30,
    'PHONE_CONF_THRESHOLD': 0.25,
    'DISTRACTED_FRAMES': 10,
    'SCREENSHOT_ENABLED': True,

    # ── Sunglasses fallback (driver's eyes are hidden) ────────────────────
    'SUNGLASSES_DARKNESS_RATIO': 0.70,   # eye-region/forehead brightness ratio below this = dark lens
    'SUNGLASSES_TEXTURE_THRESHOLD': 18.0,  # Laplacian variance below this = smooth/no visible eye detail
    'SUNGLASSES_CONFIRM_FRAMES': 5,      # frames needed to confirm sunglasses are on
    'SUNGLASSES_RELEASE_FRAMES': 8,      # frames needed to confirm sunglasses came off
    'HEAD_NOD_PITCH_THRESHOLD': 15,      # degrees of forward head-droop treated as "asleep" cue
    'HEAD_NOD_CONSEC_FRAMES': 25,        # consecutive drooped-head frames -> DROWSY (glasses mode)
           'USE_CNN_EYE': True,
       'USE_CNN_SUNGLASSES':False,
       'CNN_EYE_OPEN_MIN': 0.6,
       'CNN_SG_THRESHOLD': 0.5,
       'CNN_SG_INTERVAL': 3,
       'CNN_SG_CONFIRM': 3,
       'CNN_SG_RELEASE': 4,
}

# ─── Detector Instances ───────────────────────────────────────────────────────
face_detector = None
eye_detector = None
yawn_detector = None
head_pose_estimator = None
phone_detector = None
sunglasses_detector = None
alert_system = None
cnn_models = None
db_manager = None


def initialize_detectors():
    """Initialize all detection modules once at startup."""
    global face_detector, eye_detector, yawn_detector
    global head_pose_estimator, phone_detector, sunglasses_detector
    global alert_system, db_manager, cnn_models

    print("[INFO] Initializing detection modules...")
    face_detector = FaceDetector()
    eye_detector = EyeDetector(ear_threshold=CONFIG['EAR_THRESHOLD'])
    yawn_detector = YawnDetector(mar_threshold=CONFIG['MAR_THRESHOLD'])
    head_pose_estimator = HeadPoseEstimator()
    phone_detector = PhoneDetector(
        model_path=(os.path.join(BASE_DIR, 'yolov8_model', 'phone_detector_best.pt')
                    if os.path.exists(os.path.join(BASE_DIR, 'yolov8_model', 'phone_detector_best.pt'))
                    else os.path.join(BASE_DIR, 'yolov8_model', 'yolov8n.pt')),
        conf_threshold=CONFIG['PHONE_CONF_THRESHOLD']
    )
    if phone_detector.model is None:
        print("\n" + "!" * 60)
        print("[ERROR] Phone detection model could NOT be loaded. Phone detection is DISABLED.")
        print("        Check that yolov8_model/yolov8n.pt exists and 'ultralytics' is installed.")
        print("!" * 60 + "\n")
    sunglasses_detector = SunglassesDetector(
        darkness_ratio_threshold=CONFIG['SUNGLASSES_DARKNESS_RATIO'],
        texture_threshold=CONFIG['SUNGLASSES_TEXTURE_THRESHOLD'],
        confirm_frames=CONFIG['SUNGLASSES_CONFIRM_FRAMES'],
        release_frames=CONFIG['SUNGLASSES_RELEASE_FRAMES'],
    )
    cnn_models = CNNModels(BASE_DIR)
    alert_system = AlertSystem(audio_dir=app.config['ALERT_AUDIO_DIR'])
    db_manager = DatabaseManager(app)
    print("[INFO] All modules initialized successfully.")


# ─── Video Stream Generator ───────────────────────────────────────────────────
def generate_frames():
    """
    Core generator function: captures frames, runs detections,
    updates global driver_state, yields MJPEG frames.
    """
    global camera, detection_active, driver_state, current_session_id

    prev_time = time.time()
    drowsy_counter = 0
    yawn_counter = 0
    distracted_counter = 0
    head_nod_counter = 0     # fallback "asleep" counter used when sunglasses hide the eyes
    sg_state = False
    sg_pos = 0
    sg_neg = 0
    frame_count = 0          # for batched total_frames DB update

    with camera_lock:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        cap.release()
        return

    # Start each generator with a fresh session identifier.
    current_session_id = None

    try:
        # Start a new session
        with app.app_context():
            session = db_manager.create_session()
            current_session_id = session.id
            driver_state['session_start'] = datetime.now().isoformat()

        while detection_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # ── FPS calculation ──────────────────────────────────────────────
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time
            driver_state['fps'] = round(fps, 1)
            frame_count += 1

            # Flush total_frames to DB every 100 frames (avoid per-frame commits)
            if frame_count % 100 == 0 and current_session_id:
                with app.app_context():
                    sess_row = db_manager.get_session(current_session_id)
                    if sess_row:
                        sess_row.total_frames = (sess_row.total_frames or 0) + 100
                        db.session.commit()

            # ── Face Detection ───────────────────────────────────────────────
            wearing_sunglasses = False  # default each frame; set below if a face is found
            face_results = face_detector.detect(frame)
            driver_state['face_detected'] = face_results['face_found']

            if face_results['face_found']:
                landmarks = face_results['landmarks']

                # ── Sunglasses / Dark Eyewear Detection ────────────────────────
                # Checked before trusting EAR, since MediaPipe will still guess
                # eye-contour points even when the eyes are fully hidden behind
                # tinted or mirrored lenses, making EAR unreliable in that case.
                sg_data = sunglasses_detector.detect(frame, landmarks)
                wearing_sunglasses = sg_data['sunglasses_detected']
                if CONFIG['USE_CNN_SUNGLASSES'] and cnn_models.sunglasses_ready:
                    if frame_count % int(CONFIG['CNN_SG_INTERVAL']) == 0:
                        p_sg = cnn_models.sunglasses_prob(frame, landmarks)
                        if p_sg is not None:
                            if p_sg > CONFIG['CNN_SG_THRESHOLD']:
                                sg_pos += 1; sg_neg = 0
                            else:
                                sg_neg += 1; sg_pos = 0
                            if sg_pos >= CONFIG['CNN_SG_CONFIRM']:
                                sg_state = True
                            elif sg_neg >= CONFIG['CNN_SG_RELEASE']:
                                sg_state = False
                    wearing_sunglasses = sg_state
                driver_state['sunglasses_detected'] = wearing_sunglasses
                driver_state['detection_mode'] = 'HEAD_POSE' if wearing_sunglasses else 'EAR'

                # ── Eye Detection (EAR) ─────────────────────────────────────
                eye_data = eye_detector.detect(frame, landmarks)
                ear = eye_data['ear']
                driver_state['ear'] = round(ear, 3)

                if wearing_sunglasses:
                    # EAR can't be trusted behind dark lenses — fall back to
                    # head-pose-based "nodding off" detection instead (below).
                    driver_state['eye_status'] = 'Hidden (Sunglasses)'
                    drowsy_counter = 0
                    cv2.putText(display_frame, "SUNGLASSES DETECTED - HEAD-POSE MODE",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    head_nod_counter = 0
                eye_closed = ear < CONFIG['EAR_THRESHOLD']
                if eye_closed and CONFIG['USE_CNN_EYE'] and cnn_models.eye_ready:
                    p_open = cnn_models.eye_open_prob(frame, landmarks)
                    if p_open is not None and p_open >= CONFIG['CNN_EYE_OPEN_MIN']:
                        eye_closed = False
                if eye_closed:
                    drowsy_counter += 1
                    driver_state['eye_status'] = 'Closed'
                else:
                    drowsy_counter = 0
                    driver_state['eye_status'] = 'Open'

                    # Draw EAR on frame
                cv2.putText(display_frame, f"EAR: {ear:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ── Yawn Detection ───────────────────────────────────────────
                yawn_data = yawn_detector.detect(frame, landmarks)
                mar = yawn_data['mar']
                driver_state['mar'] = round(mar, 3)

                if mar > CONFIG['MAR_THRESHOLD']:
                    yawn_counter += 1
                else:
                    yawn_counter = 0
                # show 'Yawning' only when the mouth stays wide open (not while talking)
                driver_state['yawn_status'] = ('Yawning' if yawn_counter >= CONFIG['MAR_CONSEC_FRAMES']
                                               else 'Normal')

                driver_state['yawn_frames'] = yawn_counter

                cv2.putText(display_frame, f"MAR: {mar:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ── Head Pose Estimation ─────────────────────────────────────
                head_data = head_pose_estimator.estimate(frame, landmarks)
                pitch = head_data.get('pitch', 0.0)
                yaw = head_data.get('yaw', 0.0)
                roll = head_data.get('roll', 0.0)

                driver_state['head_pitch'] = round(pitch, 1)
                driver_state['head_yaw'] = round(yaw, 1)
                driver_state['head_roll'] = round(roll, 1)

                head_distracted = (
                    abs(pitch) > CONFIG['HEAD_PITCH_THRESHOLD'] or
                    abs(yaw) > CONFIG['HEAD_YAW_THRESHOLD']
                )

                if head_distracted:
                    driver_state['head_pose'] = 'Distracted'
                    distracted_counter += 1
                else:
                    driver_state['head_pose'] = 'Forward'

                # ── Head-Nod Fallback (sunglasses mode) ────────────────────────
                # With the eyes hidden, a sustained forward head-droop is used
                # as the drowsiness/"falling asleep" signal instead of EAR.
                if wearing_sunglasses:
                    if pitch > CONFIG['HEAD_NOD_PITCH_THRESHOLD']:
                        head_nod_counter += 1
                    else:
                        head_nod_counter = 0
                    cv2.putText(display_frame, f"Head-Nod: {head_nod_counter}/{CONFIG['HEAD_NOD_CONSEC_FRAMES']}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                driver_state['drowsy_frames'] = head_nod_counter if wearing_sunglasses else drowsy_counter

                cv2.putText(display_frame,
                            f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                            (10, 90 if not wearing_sunglasses else 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw facial landmarks
                eye_detector.draw_landmarks(display_frame, landmarks)

            else:
                # No face — possible distraction
                distracted_counter += 1
                driver_state['eye_status'] = 'Unknown'
                driver_state['yawn_status'] = 'Unknown'
                driver_state['head_pose'] = 'Unknown'
                cv2.putText(display_frame, "No Face Detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            driver_state['distracted_frames'] = distracted_counter

            # ── Phone Detection (YOLOv8) ─────────────────────────────────────
            phone_data = phone_detector.detect(frame)
            driver_state['phone_detected'] = phone_data['phone_found']

            if phone_data['phone_found']:
                distracted_counter += 1
                for box in phone_data.get('boxes', []):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, "PHONE DETECTED!",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ── Determine Overall Driver Status ──────────────────────────────
            alert_triggered = False
            alert_type = None

            drowsy_by_ear = (not wearing_sunglasses) and drowsy_counter >= CONFIG['EAR_CONSEC_FRAMES']
            drowsy_by_head_nod = wearing_sunglasses and head_nod_counter >= CONFIG['HEAD_NOD_CONSEC_FRAMES']

            if drowsy_by_ear:
                driver_state['status'] = 'DROWSY'
                alert_triggered = True
                alert_type = 'DROWSY'
                alert_msg = ("DROWSINESS ALERT! (Sunglasses detected \u2013 head-nod fallback)"
                             if drowsy_by_head_nod else "DROWSINESS ALERT!")
                _draw_alert_overlay(display_frame, alert_msg, (0, 0, 255))

            elif yawn_counter >= CONFIG['MAR_CONSEC_FRAMES']:
                driver_state['status'] = 'YAWNING'
                alert_triggered = True
                alert_type = 'YAWN'
                _draw_alert_overlay(display_frame, "YAWN DETECTED!", (0, 165, 255))

            elif drowsy_by_head_nod or distracted_counter >= CONFIG['DISTRACTED_FRAMES'] or driver_state['phone_detected']:
                driver_state['status'] = 'DISTRACTED'
                alert_triggered = True
                alert_type = 'DISTRACTED'
                _draw_alert_overlay(display_frame, "DISTRACTION ALERT! (head down, sunglasses on)" if drowsy_by_head_nod else "DISTRACTION ALERT!", (0, 165, 255))

            else:
                driver_state['status'] = 'NORMAL'

            # ── Trigger Alert ─────────────────────────────────────────────────
            if alert_triggered and alert_type:
                alert_result = alert_system.trigger(alert_type)

                # Count and save only alerts accepted by the cooldown system.
                # Detection conditions may remain true across many frames;
                # rejected attempts must not create duplicate database records.
                if alert_result.get('fired', False):
                    driver_state['alert_count'] += 1

                    # Capture screenshot (respects SCREENSHOT_ENABLED setting)
                    screenshot_path = None
                    if CONFIG.get('SCREENSHOT_ENABLED', True):
                        screenshot_path = _capture_screenshot(display_frame, alert_type)

                    # Log the accepted detection and alert
                    with app.app_context():
                        db_manager.log_detection(
                            session_id=current_session_id,
                            detection_type=alert_type,
                            ear=driver_state['ear'],
                            mar=driver_state['mar'],
                            head_pitch=driver_state['head_pitch'],
                            head_yaw=driver_state['head_yaw'],
                            screenshot_path=screenshot_path
                        )
                        db_manager.log_alert(
                            session_id=current_session_id,
                            alert_type=alert_type,
                            audio_played=alert_result.get('channel') == 'audio',
                            tts_played=alert_result.get('channel') == 'tts',
                        )

                    # Reset counters only after an alert is actually accepted.
                    if alert_type == 'DROWSY':
                        drowsy_counter = 0
                        head_nod_counter = 0
                    elif alert_type == 'YAWN':
                        yawn_counter = 0
                    elif alert_type == 'DISTRACTED':
                        distracted_counter = 0
                        head_nod_counter = 0

            # ── Draw HUD ──────────────────────────────────────────────────────
            _draw_hud(display_frame, driver_state)

            # ── Emit state via SocketIO ───────────────────────────────────────
            socketio.emit('driver_state', driver_state)

            # ── Encode and yield frame ────────────────────────────────────────
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        # Always release the webcam, including when the stream is closed.
        cap.release()

        # End the database session safely.
        with app.app_context():
            if current_session_id is not None:
                db_manager.end_session(current_session_id)
                current_session_id = None


def _draw_alert_overlay(frame, text, color):
    """Draw a prominent alert banner on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 80), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (w // 2 - len(text) * 9, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


def _draw_hud(frame, state):
    """Draw heads-up display with status indicators."""
    h, w = frame.shape[:2]

    # Status box
    status_color = {
        'NORMAL': (0, 200, 0),
        'DROWSY': (0, 0, 255),
        'YAWNING': (0, 165, 255),
        'DISTRACTED': (0, 165, 255),
    }.get(state['status'], (128, 128, 128))

    cv2.rectangle(frame, (w - 220, 10), (w - 10, 220), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 220, 10), (w - 10, 220), status_color, 2)

    lines = [
        f"Status: {state['status']}",
        f"Eyes: {state['eye_status']}",
        f"Yawn: {state['yawn_status']}",
        f"Head: {state['head_pose']}",
        f"Phone: {'YES' if state['phone_detected'] else 'No'}",
        f"Glasses: {'YES' if state['sunglasses_detected'] else 'No'}",
        f"FPS: {state['fps']}",
    ]
    for i, line in enumerate(lines):
        color = (0, 0, 255) if ('YES' in line or 'Closed' in line or 'Hidden' in line or 'DROWSY' in line) else (255, 255, 255)
        cv2.putText(frame, line, (w - 215, 35 + i * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _capture_screenshot(frame, alert_type):
    """Save a screenshot when an alert fires."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    path = os.path.join(app.config['SCREENSHOT_DIR'], filename)
    cv2.imwrite(path, frame)
    return path


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page — redirect to login if not authenticated."""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """Live monitoring dashboard."""
    return render_template('dashboard.html', config=CONFIG)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_detection', methods=['POST'])
@login_required
def start_detection():
    """Start the detection loop."""
    global detection_active
    if not detection_active:
        detection_active = True
        return jsonify({'status': 'started', 'message': 'Detection started'})
    return jsonify({'status': 'already_running', 'message': 'Detection already active'})


@app.route('/stop_detection', methods=['POST'])
@login_required
def stop_detection():
    """Stop the detection loop."""
    global detection_active, driver_state
    detection_active = False
    driver_state['status'] = 'NORMAL'
    driver_state['alert_count'] = 0
    return jsonify({'status': 'stopped', 'message': 'Detection stopped'})


@app.route('/logs')
@login_required
def logs():
    """Detection logs page."""
    with app.app_context():
        logs_data = db_manager.get_recent_logs(limit=100)
    return render_template('logs.html', logs=logs_data)


@app.route('/reports')
@login_required
def reports():
    """Reports and analytics page."""
    with app.app_context():
        stats = db_manager.get_statistics()
    return render_template('reports.html', stats=stats)


@app.route('/api/stats')
@login_required
def api_stats():
    """JSON endpoint for dashboard charts."""
    with app.app_context():
        stats = db_manager.get_statistics()
    return jsonify(stats)


# Make login_required return JSON for API routes instead of HTML redirect
@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith('/api/'):
        return jsonify({'error': 'unauthorized'}), 401
    return redirect(url_for('login', next=request.url))


@app.route('/api/logs')
@login_required
def api_logs():
    """JSON endpoint for recent detection logs."""
    limit = request.args.get('limit', 50, type=int)
    with app.app_context():
        logs_data = db_manager.get_recent_logs(limit=limit)
    return jsonify([log.to_dict() for log in logs_data])


@app.route('/api/driver_state')
@login_required
def api_driver_state():
    """Current driver state as JSON (polling fallback)."""
    return jsonify(driver_state)


@app.route('/api/config', methods=['GET', 'POST'])
@login_required
def api_config():
    """Get or update detection thresholds."""
    global CONFIG
    if request.method == 'POST':
        data = request.get_json()
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = float(value)
                if key == 'EAR_THRESHOLD' and eye_detector:
                    eye_detector.ear_threshold = float(value)
                if key == 'MAR_THRESHOLD' and yawn_detector:
                    yawn_detector.mar_threshold = float(value)
                if key == 'SUNGLASSES_DARKNESS_RATIO' and sunglasses_detector:
                    sunglasses_detector.darkness_ratio_threshold = float(value)
                if key == 'SUNGLASSES_TEXTURE_THRESHOLD' and sunglasses_detector:
                    sunglasses_detector.texture_threshold = float(value)
        return jsonify({'status': 'updated', 'config': CONFIG})
    return jsonify(CONFIG)


@app.route('/api/screenshots')
@login_required
def api_screenshots():
    """List recent screenshots."""
    screenshot_dir = app.config['SCREENSHOT_DIR']
    files = sorted(
        [f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')],
        reverse=True
    )[:20]
    return jsonify({'screenshots': files})


@app.route('/api/clear_logs', methods=['POST'])
@login_required
def api_clear_logs():
    """Clear all detection logs."""
    with app.app_context():
        db_manager.clear_logs()
    return jsonify({'status': 'cleared'})


@app.route('/api/test_alert', methods=['POST'])
@login_required
def api_test_alert():
    """Fire a test alert of the requested type (bypasses cooldown)."""
    data = request.get_json() or {}
    alert_type = data.get('type', 'DROWSY').upper()
    if alert_type not in ('DROWSY', 'YAWN', 'DISTRACTED', 'PHONE'):
        return jsonify({'status': 'error', 'message': 'Unknown alert type'}), 400
    result = alert_system.test_alert(alert_type)
    return jsonify({'status': 'ok', 'result': result})


@app.route('/api/alert_settings', methods=['GET', 'POST'])
@login_required
def api_alert_settings():
    """Get or update audio/TTS/screenshot alert settings."""
    if request.method == 'POST':
        data = request.get_json() or {}
        if 'audio_enabled' in data and alert_system:
            alert_system.audio_enabled = bool(data['audio_enabled'])
        if 'tts_enabled' in data and alert_system:
            alert_system.tts_enabled = bool(data['tts_enabled'])
        if 'screenshot_enabled' in data:
            CONFIG['SCREENSHOT_ENABLED'] = bool(data['screenshot_enabled'])
        return jsonify({'status': 'updated'})
    return jsonify({
        'audio_enabled':      alert_system.audio_enabled if alert_system else False,
        'tts_enabled':        alert_system.tts_enabled if alert_system else False,
        'screenshot_enabled': CONFIG.get('SCREENSHOT_ENABLED', True),
    })


@app.route('/screenshots/<path:filename>')
@login_required
def serve_screenshot(filename):
    """Serve screenshot images from the screenshots directory."""
    from flask import send_from_directory
    return send_from_directory(app.config['SCREENSHOT_DIR'], filename)


@app.route('/settings')
@login_required
def settings():
    """Settings page."""
    return render_template('settings.html', config=CONFIG)


# ─── Authentication Routes ────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        user = User.query.filter_by(username=username).first()
        if user and user.is_active and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')
        if not username or not email or not password:
            error = 'All fields are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif User.query.filter_by(username=username).first():
            error = 'Username already taken.'
        elif User.query.filter_by(email=email).first():
            error = 'Email already registered.'
        else:
            user = User(username=username, email=email, full_name=full_name)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('register.html', error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ─── SocketIO Events ──────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to drowsiness detection server'})
    emit('driver_state', driver_state)


@socketio.on('disconnect')
def handle_disconnect():
    print('[SocketIO] Client disconnected')


@socketio.on('request_state')
def handle_request_state():
    emit('driver_state', driver_state)


# ─── Application Bootstrap ────────────────────────────────────────────────────

def create_database():
    """Create all database tables and seed default admin user."""
    with app.app_context():
        db.create_all()
        # Create default admin if no users exist
        if User.query.count() == 0:
            admin = User(
                username='admin',
                email='admin@drowsyguard.com',
                full_name='System Administrator',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("[INFO] Default admin created — username: admin  password: admin123")
        print("[INFO] Database tables created.")


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(app.config['SCREENSHOT_DIR'], exist_ok=True)
    os.makedirs(app.config['ALERT_AUDIO_DIR'], exist_ok=True)


def create_placeholder_audio():
    """Create a placeholder alert audio note if pygame is unavailable."""
    alert_dir = app.config['ALERT_AUDIO_DIR']
    note_path = os.path.join(alert_dir, 'README.txt')
    if not os.path.exists(note_path):
        with open(note_path, 'w') as f:
            f.write("Place alert audio files here:\n")
            f.write("  - alert_drowsy.mp3\n")
            f.write("  - alert_yawn.mp3\n")
            f.write("  - alert_distracted.mp3\n")


if __name__ == '__main__':
    print("=" * 60)
    print("  AI Driver Drowsiness & Distraction Detection System")
    print("  Starting Flask server on http://127.0.0.1:5000")
    print("=" * 60)

    create_database()
    create_placeholder_audio()
    ensure_directories()
    initialize_detectors()

    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )
