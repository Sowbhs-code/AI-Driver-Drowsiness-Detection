"""
face_detector.py
─────────────────────────────────────────────────────────
Detects faces and extracts 468-point MediaPipe landmarks.
Returns normalised landmarks + bounding box per frame.
─────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """
    Wraps MediaPipe FaceMesh for face detection and landmark extraction.

    Attributes
    ----------
    face_mesh : mp.solutions.face_mesh.FaceMesh
    mp_drawing : mp.solutions.drawing_utils
    """

    # MediaPipe FaceMesh landmark indices for key regions
    LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LIPS_INDICES      = [61,  146, 91,  181, 84,  17,  314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
                         14,  87,  178, 88,  95,  185, 40,  39,  37,  0,   267, 269, 270, 409, 415, 310,
                         311, 312, 13,  82,  81,  42,  183, 78]
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109]

    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing   = mp.solutions.drawing_utils
        self.mp_styles    = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run face detection on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image

        Returns
        -------
        dict
            face_found  : bool
            landmarks   : list[tuple(x_px, y_px, z)]  or []
            bbox        : (x1, y1, x2, y2) in pixels  or None
            num_faces   : int
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {'face_found': False, 'landmarks': [], 'bbox': None, 'num_faces': 0}

        face_landmarks = results.multi_face_landmarks[0]

        # Convert normalised landmarks → pixel coordinates
        landmarks = [
            (int(lm.x * w), int(lm.y * h), lm.z)
            for lm in face_landmarks.landmark
        ]

        # Bounding box from all landmark x/y values
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        return {
            'face_found': True,
            'landmarks': landmarks,
            'bbox': bbox,
            'num_faces': len(results.multi_face_landmarks)
        }

    def draw_face_box(self, frame: np.ndarray, bbox: tuple,
                      status: str = 'NORMAL') -> None:
        """Draw bounding box around detected face."""
        color_map = {
            'NORMAL':     (0, 200, 0),
            'DROWSY':     (0, 0, 255),
            'YAWNING':    (0, 165, 255),
            'DISTRACTED': (0, 165, 255),
        }
        color = color_map.get(status, (255, 255, 255))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Face: {status}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def get_region_landmarks(self, landmarks: list, indices: list) -> np.ndarray:
        """Extract pixel-coordinate array for a subset of landmark indices."""
        return np.array([(landmarks[i][0], landmarks[i][1]) for i in indices], dtype=np.int32)

    def release(self):
        self.face_mesh.close()
