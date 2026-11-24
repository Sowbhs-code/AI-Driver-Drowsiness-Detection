"""
distraction_detector.py
─────────────────────────────────────────────────────────
Aggregates all individual signals (eye, yawn, head pose,
phone) into a unified distraction / drowsiness score
and overall driver status classification.
─────────────────────────────────────────────────────────
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class DriverStatus:
    """Snapshot of the current driver status."""
    status: str          = 'NORMAL'   # NORMAL | DROWSY | YAWNING | DISTRACTED
    drowsiness_score: float = 0.0     # 0.0 – 1.0
    distraction_score: float = 0.0
    alert_level: str     = 'LOW'      # LOW | MEDIUM | HIGH | CRITICAL
    details: dict        = field(default_factory=dict)


class DistractionDetector:
    """
    Fuses multiple detection signals with a rolling temporal
    window to produce a stable, noise-resistant driver status.

    Parameters
    ----------
    window_size       : int    number of recent frames to consider
    ear_threshold     : float  EAR below which eyes are 'closed'
    mar_threshold     : float  MAR above which mouth is 'yawning'
    head_pitch_thresh : float  degrees
    head_yaw_thresh   : float  degrees
    ear_consec_frames : int    consecutive closed-eye frames → drowsy alert
    mar_consec_frames : int    consecutive yawn frames → yawn alert
    """

    def __init__(self,
                 window_size: int = 30,
                 ear_threshold: float = 0.25,
                 mar_threshold: float = 0.6,
                 head_pitch_thresh: float = 20.0,
                 head_yaw_thresh: float = 30.0,
                 ear_consec_frames: int = 20,
                 mar_consec_frames: int = 15):

        self.ear_threshold     = ear_threshold
        self.mar_threshold     = mar_threshold
        self.head_pitch_thresh = head_pitch_thresh
        self.head_yaw_thresh   = head_yaw_thresh
        self.ear_consec_frames = ear_consec_frames
        self.mar_consec_frames = mar_consec_frames

        # Rolling windows (deques) for temporal smoothing
        self._ear_window:  Deque[float] = deque(maxlen=window_size)
        self._mar_window:  Deque[float] = deque(maxlen=window_size)
        self._head_window: Deque[bool]  = deque(maxlen=window_size)
        self._phone_window: Deque[bool] = deque(maxlen=window_size)
        self._face_window:  Deque[bool] = deque(maxlen=window_size)

        # Consecutive-frame counters
        self._closed_count = 0
        self._yawn_count   = 0

        # Cumulative session statistics
        self._total_frames       = 0
        self._drowsy_frames      = 0
        self._distracted_frames  = 0
        self._alert_count        = 0
        self._session_start      = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               ear: float,
               mar: float,
               head_pitch: float,
               head_yaw: float,
               phone_detected: bool,
               face_detected: bool) -> DriverStatus:
        """
        Ingest one frame's worth of sensor readings and return
        the updated DriverStatus.

        Parameters
        ----------
        ear            : float  Eye Aspect Ratio
        mar            : float  Mouth Aspect Ratio
        head_pitch     : float  degrees
        head_yaw       : float  degrees
        phone_detected : bool
        face_detected  : bool

        Returns
        -------
        DriverStatus
        """
        self._total_frames += 1

        # ── Update rolling windows ────────────────────────────────────────
        self._ear_window.append(ear)
        self._mar_window.append(mar)
        self._head_window.append(
            abs(head_pitch) > self.head_pitch_thresh or
            abs(head_yaw)   > self.head_yaw_thresh
        )
        self._phone_window.append(phone_detected)
        self._face_window.append(face_detected)

        # ── Consecutive-frame counters ────────────────────────────────────
        if face_detected and ear < self.ear_threshold:
            self._closed_count += 1
        else:
            self._closed_count = max(0, self._closed_count - 1)

        if face_detected and mar > self.mar_threshold:
            self._yawn_count += 1
        else:
            self._yawn_count = max(0, self._yawn_count - 1)

        # ── Compute scores ────────────────────────────────────────────────
        drowsiness_score  = self._compute_drowsiness_score()
        distraction_score = self._compute_distraction_score()

        # ── Classify status ───────────────────────────────────────────────
        status      = 'NORMAL'
        alert_level = 'LOW'

        if self._closed_count >= self.ear_consec_frames:
            status = 'DROWSY'
            alert_level = 'CRITICAL'
            self._drowsy_frames += 1
            self._alert_count   += 1

        elif self._yawn_count >= self.mar_consec_frames:
            status = 'YAWNING'
            alert_level = 'HIGH'
            self._drowsy_frames += 1

        elif (sum(self._head_window) / max(len(self._head_window), 1) > 0.5 or
              sum(self._phone_window) / max(len(self._phone_window), 1) > 0.3 or
              not face_detected):
            status = 'DISTRACTED'
            alert_level = 'HIGH'
            self._distracted_frames += 1

        elif drowsiness_score > 0.5:
            status = 'DROWSY'
            alert_level = 'MEDIUM'

        details = {
            'closed_count':      self._closed_count,
            'yawn_count':        self._yawn_count,
            'head_distracted':   self._head_window[-1] if self._head_window else False,
            'phone_detected':    phone_detected,
            'face_detected':     face_detected,
            'avg_ear':           sum(self._ear_window) / max(len(self._ear_window), 1),
            'avg_mar':           sum(self._mar_window) / max(len(self._mar_window), 1),
        }

        return DriverStatus(
            status=status,
            drowsiness_score=drowsiness_score,
            distraction_score=distraction_score,
            alert_level=alert_level,
            details=details,
        )

    def get_session_stats(self) -> dict:
        """Return cumulative session statistics."""
        elapsed = time.time() - self._session_start
        return {
            'total_frames':      self._total_frames,
            'drowsy_frames':     self._drowsy_frames,
            'distracted_frames': self._distracted_frames,
            'alert_count':       self._alert_count,
            'elapsed_seconds':   round(elapsed, 1),
            'drowsiness_pct':    round(self._drowsy_frames / max(self._total_frames, 1) * 100, 1),
            'distraction_pct':   round(self._distracted_frames / max(self._total_frames, 1) * 100, 1),
        }

    def reset(self) -> None:
        """Reset all counters (call at start of new session)."""
        self._closed_count = 0
        self._yawn_count   = 0
        self._total_frames = self._drowsy_frames = self._distracted_frames = self._alert_count = 0
        self._session_start = time.time()
        for w in (self._ear_window, self._mar_window,
                  self._head_window, self._phone_window, self._face_window):
            w.clear()

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _compute_drowsiness_score(self) -> float:
        """
        Weighted combination of EAR deficit and consecutive-close count.
        Returns a value in [0, 1].
        """
        if not self._ear_window:
            return 0.0
        avg_ear   = sum(self._ear_window) / len(self._ear_window)
        ear_score = max(0.0, (self.ear_threshold - avg_ear) / (self.ear_threshold + 1e-9))
        seq_score = min(1.0, self._closed_count / (self.ear_consec_frames + 1e-9))
        return float(min(1.0, 0.4 * ear_score + 0.6 * seq_score))

    def _compute_distraction_score(self) -> float:
        """
        Weighted combination of head movement ratio and phone detection ratio.
        Returns a value in [0, 1].
        """
        n = max(len(self._head_window), 1)
        head_ratio  = sum(self._head_window)  / n
        phone_ratio = sum(self._phone_window) / max(len(self._phone_window), 1)
        face_ratio  = 1.0 - (sum(self._face_window) / max(len(self._face_window), 1))
        return float(min(1.0, 0.4 * head_ratio + 0.4 * phone_ratio + 0.2 * face_ratio))
