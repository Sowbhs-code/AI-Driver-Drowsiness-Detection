"""
yawn_detector.py
─────────────────────────────────────────────────────────
Mouth Aspect Ratio (MAR) based yawn detection.

MAR formula (analogous to EAR):
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)

MAR > 0.6  → mouth wide open / yawning
─────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist


class YawnDetector:
    """
    Detects yawning using the Mouth Aspect Ratio (MAR)
    computed from MediaPipe 468-point facial landmarks.
    """

    # MediaPipe landmark indices for mouth (8-point MAR subset)
    # p1(left-corner), p2..p4(top), p5(right-corner), p6..p8(bottom). INNER lip points: a closed mouth gives MAR near 0
    MOUTH_MAR_IDX = [78, 82, 13, 312, 308, 317, 14, 87]   # INNER lip: corners 78/308, top 82/13/312, bottom 87/14/317

    # Full outer lip contour for drawing
    OUTER_LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                     291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                     95, 185, 40, 39, 37, 0, 267, 269, 270, 409]

    # Inner lip contour
    INNER_LIP_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     415, 310, 311, 312, 13, 82, 81, 42, 183]

    def __init__(self, mar_threshold: float = 0.6):
        self.mar_threshold = mar_threshold

    # ── Core MAR Calculation ──────────────────────────────────────────────────

    @staticmethod
    def _mar(mouth_pts: np.ndarray) -> float:
        """
        Compute Mouth Aspect Ratio for 8 mouth landmark points.

        Parameters
        ----------
        mouth_pts : np.ndarray  shape (8, 2)
            [left-corner, top-outer-L, top-mid, top-outer-R,
             right-corner, bot-outer-R, bot-mid, bot-outer-L]

        Returns
        -------
        float  Mouth Aspect Ratio
        """
        # Vertical distances (3 pairs)
        A = dist.euclidean(mouth_pts[1], mouth_pts[7])  # top-outer-L to bot-outer-L
        B = dist.euclidean(mouth_pts[2], mouth_pts[6])  # top-mid to bot-mid
        C = dist.euclidean(mouth_pts[3], mouth_pts[5])  # top-outer-R to bot-outer-R
        # Horizontal distance
        D = dist.euclidean(mouth_pts[0], mouth_pts[4])  # left-corner to right-corner
        return (A + B + C) / (2.0 * D + 1e-9)

    def _get_mouth_points(self, landmarks: list) -> np.ndarray:
        return np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in self.MOUTH_MAR_IDX],
            dtype=np.float64
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, landmarks: list) -> dict:
        """
        Compute MAR and classify mouth state.

        Parameters
        ----------
        frame     : np.ndarray  BGR image
        landmarks : list of (x_px, y_px, z) from FaceDetector

        Returns
        -------
        dict
            mar          : float
            yawning      : bool
            mouth_pts    : np.ndarray (8,2)
            mouth_height : float   vertical opening in pixels
            mouth_width  : float   horizontal opening in pixels
        """
        mouth_pts    = self._get_mouth_points(landmarks)
        mar          = self._mar(mouth_pts)
        mouth_height = dist.euclidean(mouth_pts[2], mouth_pts[6])
        mouth_width  = dist.euclidean(mouth_pts[0], mouth_pts[4])

        return {
            'mar': mar,
            'yawning': mar > self.mar_threshold,
            'mouth_pts': mouth_pts,
            'mouth_height': mouth_height,
            'mouth_width': mouth_width,
        }

    def draw_landmarks(self, frame: np.ndarray, landmarks: list,
                       yawning: bool = False) -> None:
        """Draw mouth contours on the frame."""
        color = (0, 0, 255) if yawning else (0, 255, 0)

        outer = np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in self.OUTER_LIP_IDX],
            dtype=np.int32
        )
        inner = np.array(
            [(landmarks[i][0], landmarks[i][1]) for i in self.INNER_LIP_IDX],
            dtype=np.int32
        )
        cv2.polylines(frame, [outer], True, color, 1)
        cv2.polylines(frame, [inner], True, (0, 200, 200), 1)

        # MAR landmark dots
        for i in self.MOUTH_MAR_IDX:
            x, y = landmarks[i][0], landmarks[i][1]
            cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

    def mar_to_percentage(self, mar: float,
                          closed_mar: float = 0.2,
                          open_mar: float = 1.0) -> float:
        """Convert MAR to a mouth-openness percentage (0–100%)."""
        pct = (mar - closed_mar) / (open_mar - closed_mar + 1e-9) * 100
        return float(np.clip(pct, 0.0, 100.0))