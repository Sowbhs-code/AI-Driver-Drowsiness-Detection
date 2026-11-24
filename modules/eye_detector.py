"""
eye_detector.py
─────────────────────────────────────────────────────────
Eye Aspect Ratio (EAR) based drowsiness detection.

EAR formula (Soukupová & Čech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

EAR ≈ 0.30 → eyes open
EAR < 0.25 → eyes closed / drowsy
─────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist


class EyeDetector:
    """
    Computes Eye Aspect Ratio from MediaPipe 468-point landmarks
    and classifies eye state (open / closed).
    """

    # MediaPipe landmark indices for left and right eyes (6-point EAR subset)
    # Format: [p1(outer), p2(top-outer), p3(top-inner), p4(inner), p5(bot-inner), p6(bot-outer)]
    LEFT_EYE_EAR_IDX  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR_IDX = [33,  160, 158, 133, 153, 144]

    # Full eye contour for drawing
    LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def __init__(self, ear_threshold: float = 0.25):
        self.ear_threshold = ear_threshold

    # ── Core EAR Calculation ──────────────────────────────────────────────────

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        """
        Compute EAR for 6 eye landmark points.

        Parameters
        ----------
        eye_pts : np.ndarray  shape (6, 2)

        Returns
        -------
        float  Eye Aspect Ratio
        """
        # Vertical distances
        A = dist.euclidean(eye_pts[1], eye_pts[5])
        B = dist.euclidean(eye_pts[2], eye_pts[4])
        # Horizontal distance
        C = dist.euclidean(eye_pts[0], eye_pts[3])
        return (A + B) / (2.0 * C + 1e-9)

    def _get_eye_points(self, landmarks: list, indices: list) -> np.ndarray:
        """Extract 2-D pixel coords for given landmark indices."""
        return np.array([(landmarks[i][0], landmarks[i][1]) for i in indices],
                        dtype=np.float64)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, landmarks: list) -> dict:
        """
        Compute EAR and classify eye state.

        Parameters
        ----------
        frame     : np.ndarray  BGR image (unused for calc, kept for API consistency)
        landmarks : list of (x_px, y_px, z) tuples from FaceDetector

        Returns
        -------
        dict
            ear             : float   average EAR of both eyes
            left_ear        : float
            right_ear       : float
            eyes_closed     : bool
            left_eye_pts    : np.ndarray (6,2)
            right_eye_pts   : np.ndarray (6,2)
        """
        left_pts  = self._get_eye_points(landmarks, self.LEFT_EYE_EAR_IDX)
        right_pts = self._get_eye_points(landmarks, self.RIGHT_EYE_EAR_IDX)

        left_ear  = self._ear(left_pts)
        right_ear = self._ear(right_pts)
        avg_ear   = (left_ear + right_ear) / 2.0

        return {
            'ear': avg_ear,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'eyes_closed': avg_ear < self.ear_threshold,
            'left_eye_pts': left_pts,
            'right_eye_pts': right_pts,
        }

    def draw_landmarks(self, frame: np.ndarray, landmarks: list) -> None:
        """
        Draw eye contours and EAR value on the frame.

        Parameters
        ----------
        frame     : np.ndarray  BGR image (modified in-place)
        landmarks : list of (x_px, y_px, z) tuples
        """
        # Left eye contour
        left_contour  = np.array([(landmarks[i][0], landmarks[i][1])
                                   for i in self.LEFT_EYE_CONTOUR], dtype=np.int32)
        right_contour = np.array([(landmarks[i][0], landmarks[i][1])
                                   for i in self.RIGHT_EYE_CONTOUR], dtype=np.int32)

        cv2.polylines(frame, [left_contour],  True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_contour], True, (0, 255, 0), 1)

        # EAR landmark dots
        for idx in self.LEFT_EYE_EAR_IDX + self.RIGHT_EYE_EAR_IDX:
            x, y = landmarks[idx][0], landmarks[idx][1]
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    def get_eye_region(self, frame: np.ndarray, landmarks: list,
                       eye: str = 'left', padding: int = 10) -> np.ndarray:
        """
        Crop and return the eye region as a small BGR patch.

        Parameters
        ----------
        eye : 'left' | 'right'
        """
        h, w = frame.shape[:2]
        indices = self.LEFT_EYE_CONTOUR if eye == 'left' else self.RIGHT_EYE_CONTOUR
        pts = np.array([(landmarks[i][0], landmarks[i][1]) for i in indices])
        x1 = max(0, pts[:, 0].min() - padding)
        y1 = max(0, pts[:, 1].min() - padding)
        x2 = min(w, pts[:, 0].max() + padding)
        y2 = min(h, pts[:, 1].max() + padding)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def ear_to_percentage(ear: float, open_ear: float = 0.35,
                           closed_ear: float = 0.15) -> float:
        """Convert EAR to an openness percentage (0–100%)."""
        pct = (ear - closed_ear) / (open_ear - closed_ear + 1e-9) * 100
        return float(np.clip(pct, 0.0, 100.0))
