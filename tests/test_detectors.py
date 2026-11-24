"""
test_detectors.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for all detection modules.
Run: pytest tests/test_detectors.py -v
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2

# ── Helpers ───────────────────────────────────────────────────────────────────

def blank_frame(h=480, w=640):
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)

def fake_landmarks(n=468):
    """Generate synthetic landmarks with realistic eye geometry."""
    np.random.seed(0)
    lm = [
        (np.random.randint(100, 540),
         np.random.randint(80, 400),
         np.random.uniform(-0.1, 0.1))
        for _ in range(n)
    ]

    # Left eye: six EAR points
    left_eye = {
        362: (350, 200), 385: (360, 196),
        387: (375, 196), 263: (390, 200),
        373: (375, 204), 380: (360, 204)
    }

    # Right eye: six EAR points
    right_eye = {
        33: (200, 200), 160: (210, 196),
        158: (225, 196), 133: (240, 200),
        153: (225, 204), 144: (210, 204)
    }

    for index, point in {**left_eye, **right_eye}.items():
        lm[index] = (point[0], point[1], 0.0)

    return lm

# ── Eye Detector Tests ────────────────────────────────────────────────────────

class TestEyeDetector:
    def setup_method(self):
        from modules.eye_detector import EyeDetector
        self.det = EyeDetector(ear_threshold=0.25)
        self.lm  = fake_landmarks()
        self.frame = blank_frame()

    def test_instantiation(self):
        assert self.det is not None
        assert self.det.ear_threshold == 0.25

    def test_detect_returns_dict(self):
        result = self.det.detect(self.frame, self.lm)
        assert isinstance(result, dict)

    def test_detect_has_required_keys(self):
        result = self.det.detect(self.frame, self.lm)
        for key in ('ear', 'left_ear', 'right_ear', 'eyes_closed'):
            assert key in result, f"Missing key: {key}"

    def test_ear_range(self):
        result = self.det.detect(self.frame, self.lm)
        assert 0.0 <= result['ear'] <= 1.0, f"EAR out of range: {result['ear']}"

    def test_eyes_closed_flag_below_threshold(self):
        from modules.eye_detector import EyeDetector
        det = EyeDetector(ear_threshold=0.99)   # almost always closed
        result = det.detect(self.frame, self.lm)
        assert isinstance(result['eyes_closed'], (bool, np.bool_))

    def test_ear_to_percentage(self):
        pct = self.det.ear_to_percentage(0.30)
        assert 0.0 <= pct <= 100.0

    def test_draw_landmarks_no_exception(self):
        """draw_landmarks must not raise on a valid frame + landmarks."""
        self.det.draw_landmarks(self.frame, self.lm)

    def test_threshold_setter(self):
        self.det.ear_threshold = 0.30
        assert self.det.ear_threshold == 0.30


# ── Yawn Detector Tests ───────────────────────────────────────────────────────

class TestYawnDetector:
    def setup_method(self):
        from modules.yawn_detector import YawnDetector
        self.det   = YawnDetector(mar_threshold=0.6)
        self.lm    = fake_landmarks()
        self.frame = blank_frame()

    def test_instantiation(self):
        assert self.det.mar_threshold == 0.6

    def test_detect_returns_dict(self):
        result = self.det.detect(self.frame, self.lm)
        assert isinstance(result, dict)

    def test_detect_keys(self):
        result = self.det.detect(self.frame, self.lm)
        for key in ('mar', 'yawning', 'mouth_height', 'mouth_width'):
            assert key in result

    def test_mar_non_negative(self):
        result = self.det.detect(self.frame, self.lm)
        assert result['mar'] >= 0.0

    def test_yawning_is_bool(self):
        result = self.det.detect(self.frame, self.lm)
        assert isinstance(result['yawning'], (bool, np.bool_))

    def test_mar_to_percentage(self):
        pct = self.det.mar_to_percentage(0.6)
        assert 0.0 <= pct <= 100.0


# ── Head Pose Estimator Tests ─────────────────────────────────────────────────

class TestHeadPoseEstimator:
    def setup_method(self):
        from modules.head_pose import HeadPoseEstimator
        self.est   = HeadPoseEstimator()
        self.lm    = fake_landmarks()
        self.frame = blank_frame()

    def test_instantiation(self):
        assert self.est is not None

    def test_estimate_returns_dict(self):
        result = self.est.estimate(self.frame, self.lm)
        assert isinstance(result, dict)

    def test_estimate_has_angle_keys(self):
        result = self.est.estimate(self.frame, self.lm)
        for key in ('pitch', 'yaw', 'roll', 'success'):
            assert key in result

    def test_angles_are_floats(self):
        result = self.est.estimate(self.frame, self.lm)
        if result['success']:
            for ang in ('pitch', 'yaw', 'roll'):
                assert isinstance(result[ang], float)

    def test_classify_direction_forward(self):
        direction = self.est.classify_head_direction(0.0, 0.0)
        assert direction == 'Forward'

    def test_classify_direction_left(self):
        direction = self.est.classify_head_direction(0.0, -40.0)
        assert 'Left' in direction

    def test_classify_direction_down(self):
        direction = self.est.classify_head_direction(25.0, 0.0)
        assert 'Down' in direction


# ── Phone Detector Tests ──────────────────────────────────────────────────────

class TestPhoneDetector:
    def setup_method(self):
        from modules.phone_detector import PhoneDetector
        self.det   = PhoneDetector(model_path='yolov8n.pt', conf_threshold=0.5)
        self.frame = blank_frame()

    def test_instantiation(self):
        assert self.det is not None

    def test_detect_returns_dict(self):
        result = self.det.detect(self.frame)
        assert isinstance(result, dict)

    def test_detect_keys(self):
        result = self.det.detect(self.frame)
        for key in ('phone_found', 'boxes', 'confidences', 'count'):
            assert key in result

    def test_blank_frame_no_phone(self):
        result = self.det.detect(self.frame)
        assert isinstance(result['phone_found'], bool)
        assert isinstance(result['boxes'], list)

    def test_confidence_threshold_setter(self):
        self.det.set_confidence_threshold(0.7)
        assert self.det.conf_threshold == 0.7

    def test_confidence_clamp(self):
        self.det.set_confidence_threshold(2.0)   # above max
        assert self.det.conf_threshold <= 1.0
        self.det.set_confidence_threshold(-1.0)  # below min
        assert self.det.conf_threshold >= 0.1


# ── Face Detector Tests ───────────────────────────────────────────────────────

class TestFaceDetector:
    def setup_method(self):
        from modules.face_detector import FaceDetector
        self.det   = FaceDetector()
        self.frame = blank_frame()

    def test_instantiation(self):
        assert self.det is not None

    def test_detect_blank_frame(self):
        result = self.det.detect(self.frame)
        assert isinstance(result, dict)
        assert 'face_found' in result
        # Blank frame should have no face
        assert result['face_found'] == False

    def test_detect_returns_landmarks_list(self):
        result = self.det.detect(self.frame)
        assert isinstance(result['landmarks'], list)

    def test_detect_returns_num_faces(self):
        result = self.det.detect(self.frame)
        assert isinstance(result['num_faces'], int)
        assert result['num_faces'] >= 0

    def test_get_region_landmarks(self):
        lm = fake_landmarks()
        pts = self.det.get_region_landmarks(lm, [0, 1, 2, 3])
        assert pts.shape == (4, 2)

    def teardown_method(self):
        self.det.release()


# ── Distraction Detector Tests ────────────────────────────────────────────────

class TestDistractionDetector:
    def setup_method(self):
        from modules.distraction_detector import DistractionDetector
        self.det = DistractionDetector(window_size=10, ear_consec_frames=5, mar_consec_frames=4)

    def test_normal_state(self):
        for _ in range(5):
            status = self.det.update(ear=0.35, mar=0.2, head_pitch=0, head_yaw=0,
                                     phone_detected=False, face_detected=True)
        assert status.status == 'NORMAL'

    def test_drowsy_state_after_consecutive_frames(self):
        for _ in range(10):
            status = self.det.update(ear=0.18, mar=0.2, head_pitch=0, head_yaw=0,
                                     phone_detected=False, face_detected=True)
        assert status.status == 'DROWSY'

    def test_distracted_phone(self):
        for _ in range(5):
            status = self.det.update(ear=0.35, mar=0.2, head_pitch=0, head_yaw=0,
                                     phone_detected=True, face_detected=True)
        assert status.status == 'DISTRACTED'

    def test_score_range(self):
        status = self.det.update(0.30, 0.3, 5.0, 10.0, False, True)
        assert 0.0 <= status.drowsiness_score  <= 1.0
        assert 0.0 <= status.distraction_score <= 1.0

    def test_session_stats(self):
        for _ in range(3):
            self.det.update(0.30, 0.3, 0, 0, False, True)
        stats = self.det.get_session_stats()
        assert stats['total_frames'] == 3
        assert 'elapsed_seconds' in stats

    def test_reset(self):
        self.det.update(0.10, 0.8, 30, 40, True, False)
        self.det.reset()
        stats = self.det.get_session_stats()
        assert stats['total_frames'] == 0
