"""
sunglasses_detector.py
─────────────────────────────────────────────────────────
Detects whether the driver is wearing sunglasses / dark
eyewear using a lightweight, model-free heuristic built on
top of the existing MediaPipe FaceMesh landmarks — no extra
ML model or training data required.

Why this is needed
-------------------
EyeDetector's EAR (Eye Aspect Ratio) relies on MediaPipe
estimating landmark points *around* the eyelid. When a
driver wears sunglasses, MediaPipe usually still reports a
face and "guesses" plausible eye-contour points, but it
cannot see the eyelid/iris — so the EAR value becomes noisy
and can no longer be trusted to tell open vs closed eyes.

Heuristic
---------
For each eye we crop the eye-contour region and compare it
against a nearby bare-skin patch (forehead):

1. Darkness ratio — sunglass lenses are almost always
   noticeably darker than the surrounding skin, even under
   varied lighting, so we compare mean pixel intensity
   (eye region) / (forehead region).
2. Texture/variance — a real, uncovered eye has high local
   contrast (sclera, iris, eyelashes, specular highlights).
   A lens — tinted or mirrored — smooths most of that out,
   so we measure Laplacian variance ("texture energy") in
   the eye region.

Both signals must agree (dark AND low-texture) before a
frame is flagged, and a short rolling counter is used to
require the condition to hold for several consecutive
frames — this avoids false positives from motion blur,
blinking, or a single bad frame.
─────────────────────────────────────────────────────────
"""

import cv2
import numpy as np


class SunglassesDetector:
    """
    Frame-by-frame + temporally-smoothed sunglasses detector.

    Parameters
    ----------
    darkness_ratio_threshold : float
        eye_brightness / forehead_brightness below this value
        is considered "suspiciously dark" (sunglasses-like).
    texture_threshold : float
        Laplacian variance below this value is considered
        "too smooth" to be a bare, visible eye.
    confirm_frames : int
        Number of consecutive positive frames required before
        `sunglasses_detected` flips to True (debounces flicker).
    release_frames : int
        Number of consecutive negative frames required before
        flipping back to False (keeps the flag stable once set).
    """

    # Reuses the same MediaPipe FaceMesh index sets as FaceDetector/EyeDetector
    LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                         388, 387, 386, 385, 384, 398]
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                         157, 158, 159, 160, 161, 246]

    # A small patch of bare forehead skin, just above the eyebrows —
    # rarely occluded by sunglasses, caps, or hair for a driver facing camera.
    FOREHEAD_INDICES = [10, 108, 151, 337]

    def __init__(self,
                 darkness_ratio_threshold: float = 0.70,
                 texture_threshold: float = 18.0,
                 confirm_frames: int = 5,
                 release_frames: int = 8):
        self.darkness_ratio_threshold = darkness_ratio_threshold
        self.texture_threshold = texture_threshold
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames

        self._pos_streak = 0
        self._neg_streak = 0
        self._locked_on = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _crop(gray: np.ndarray, pts: np.ndarray, padding: int = 6):
        h, w = gray.shape[:2]
        x1 = max(0, int(pts[:, 0].min()) - padding)
        y1 = max(0, int(pts[:, 1].min()) - padding)
        x2 = min(w, int(pts[:, 0].max()) + padding)
        y2 = min(h, int(pts[:, 1].max()) + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return gray[y1:y2, x1:x2]

    @classmethod
    def _region_stats(cls, gray: np.ndarray, landmarks: list, indices: list,
                       padding: int = 2):
        pts = np.array([(landmarks[i][0], landmarks[i][1]) for i in indices])
        crop = cls._crop(gray, pts, padding)
        if crop is None or crop.size == 0:
            return None, None
        # Light denoise so webcam sensor noise doesn't masquerade as "texture"
        if crop.shape[0] >= 3 and crop.shape[1] >= 3:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
        brightness = float(crop.mean())

        # Texture is measured on a slightly inset sub-crop so the hard
        # boundary between the (bright) surrounding skin and the (dark)
        # lens/eye region doesn't itself get counted as "texture" —
        # we only care about detail *inside* the eye, not its outline.
        ch, cw = crop.shape[:2]
        inset_y, inset_x = max(1, ch // 6), max(1, cw // 6)
        if ch - 2 * inset_y > 2 and cw - 2 * inset_x > 2:
            texture_crop = crop[inset_y:ch - inset_y, inset_x:cw - inset_x]
        else:
            texture_crop = crop
        texture = float(cv2.Laplacian(texture_crop, cv2.CV_64F).var())
        return brightness, texture

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, landmarks: list) -> dict:
        """
        Analyse the current frame for sunglasses / dark eyewear.

        Returns
        -------
        dict
            sunglasses_detected : bool   (debounced, stable flag)
            raw_positive        : bool   (this frame only, before debouncing)
            darkness_ratio      : float
            eye_texture         : float
            confidence          : float  0–1, how strongly this frame matches
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        left_b, left_t = self._region_stats(gray, landmarks, self.LEFT_EYE_INDICES)
        right_b, right_t = self._region_stats(gray, landmarks, self.RIGHT_EYE_INDICES)
        forehead_b, _ = self._region_stats(gray, landmarks, self.FOREHEAD_INDICES, padding=8)

        if left_b is None or right_b is None or forehead_b is None or forehead_b < 1e-3:
            # Couldn't measure this frame — hold last stable state.
            return {
                'sunglasses_detected': self._locked_on,
                'raw_positive': False,
                'darkness_ratio': 1.0,
                'eye_texture': 999.0,
                'confidence': 0.0,
            }

        eye_brightness = (left_b + right_b) / 2.0
        eye_texture = (left_t + right_t) / 2.0
        darkness_ratio = eye_brightness / (forehead_b + 1e-6)

        is_dark = darkness_ratio < self.darkness_ratio_threshold
        is_smooth = eye_texture < self.texture_threshold
        raw_positive = bool(is_dark and is_smooth)

        # ── Temporal debouncing (avoid flicker on a single noisy frame) ──────
        if raw_positive:
            self._pos_streak += 1
            self._neg_streak = 0
        else:
            self._neg_streak += 1
            self._pos_streak = 0

        if not self._locked_on and self._pos_streak >= self.confirm_frames:
            self._locked_on = True
        elif self._locked_on and self._neg_streak >= self.release_frames:
            self._locked_on = False

        # Rough confidence score for UI/debugging purposes
        dark_score = float(np.clip(
            (self.darkness_ratio_threshold - darkness_ratio) / self.darkness_ratio_threshold, 0, 1))
        smooth_score = float(np.clip(
            (self.texture_threshold - eye_texture) / self.texture_threshold, 0, 1))
        confidence = round((dark_score + smooth_score) / 2.0, 2)

        return {
            'sunglasses_detected': self._locked_on,
            'raw_positive': raw_positive,
            'darkness_ratio': round(darkness_ratio, 3),
            'eye_texture': round(eye_texture, 2),
            'confidence': confidence,
        }

    def reset(self):
        """Clear the debounce state (e.g. between sessions)."""
        self._pos_streak = 0
        self._neg_streak = 0
        self._locked_on = False
