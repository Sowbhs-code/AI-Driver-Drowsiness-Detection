"""
head_pose.py  (robust version)
Head pose (pitch / yaw / roll) from MediaPipe face landmarks with cv2.solvePnP.

What is different from the original:
  * The 3-D face model is MediaPipe's OWN canonical face (12 landmarks), in OpenCV camera
    convention (x right, y down, z away from the camera), units = mm.
  * It does not depend on the caller knowing whether the frame was mirrored (cv2.flip):
    every frame BOTH hypotheses are solved and the one that fits the landmarks better is used.
  * A pose is accepted only if the face is in front of the camera and fits the landmarks well.
    Otherwise success=False and angles are 0 (so a bad fit can never cause a false 'Distracted').
  * Angles are wrapped to -90..+90.  pitch: + = chin DOWN, - = head UP.

Set the environment variable DROWSY_DEBUG=1 to print a diagnostic line every 15 frames.
"""

import math
import os
import cv2
import numpy as np


class HeadPoseEstimator:

    MP_INDICES = [1, 199, 33, 263, 61, 291, 133, 362, 168, 234, 454, 152]

    # nose tip, chin, outer eye corners, mouth corners, inner eye corners, nose bridge, cheeks, chin tip
    MODEL_POINTS_3D = np.array([[0.0, 0.0, 0.0], [0.0, 68.2, 22.9], [-44.5, -37.9, 43.0], [44.5, -37.9, 43.0], [-24.6, 32.2, 31.9], [24.6, 32.2, 31.9], [-18.6, -37.1, 37.2], [18.6, -37.1, 37.2], [0.0, -44.0, 22.4], [-76.6, -18.0, 99.1], [76.6, -18.0, 99.1], [0.0, 82.8, 32.1]], dtype=np.float64)

    MAX_FIT_ERROR = 0.12      # allowed mean reprojection error as a fraction of the eye-to-eye pixel distance

    def __init__(self, mirrored=None):
        """mirrored is ignored (kept for compatibility): the mirror state is detected automatically."""
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._mirror_model = self.MODEL_POINTS_3D * np.array([-1.0, 1.0, 1.0])
        self._n = 0
        self.last_debug = {}

    def _build_camera_matrix(self, frame_shape):
        h, w = frame_shape[:2]
        focal = float(w)
        return np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)

    def _solve(self, model, pts2d):
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(model, pts2d, self._camera_matrix, self._dist_coeffs,
                                      rvec0, tvec0, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            return None
        proj, _ = cv2.projectPoints(model, rvec, tvec, self._camera_matrix, self._dist_coeffs)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)))
        return err, rvec, tvec

    def estimate(self, frame, landmarks):
        if self._camera_matrix is None or self._camera_matrix[0, 2] != frame.shape[1] / 2.0 \
                or self._camera_matrix[1, 2] != frame.shape[0] / 2.0:
            self._camera_matrix = self._build_camera_matrix(frame.shape)

        pts2d = np.array([(landmarks[i][0], landmarks[i][1]) for i in self.MP_INDICES], dtype=np.float64)
        eye_dist = float(np.linalg.norm(pts2d[2] - pts2d[3])) + 1e-9

        best = None
        for name, model in (("normal", self.MODEL_POINTS_3D), ("mirrored", self._mirror_model)):
            res = self._solve(model, pts2d)
            if res is not None and (best is None or res[0] < best[0]):
                best = (res[0], res[1], res[2], name, model)

        self._n += 1
        fail = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'rvec': None, 'tvec': None, 'success': False}
        if best is None or best[0] / eye_dist > self.MAX_FIT_ERROR:
            self.last_debug = {'fit': None if best is None else round(best[0] / eye_dist, 3), 'accepted': False}
            if os.environ.get("DROWSY_DEBUG") and self._n % 15 == 0:
                print("[POSE] rejected, fit error =", self.last_debug['fit'])
            return fail

        err, rvec, tvec, name, model = best
        rmat, _ = cv2.Rodrigues(rvec)
        x_deg, yaw, roll = self._rotation_matrix_to_euler(rmat)
        pitch = -x_deg
        pitch = (pitch + 90.0) % 180.0 - 90.0
        roll = (roll + 90.0) % 180.0 - 90.0
        self._rvec_model = model
        self.last_debug = {'hyp': name, 'fit': round(err / eye_dist, 3), 'tz_mm': round(float(tvec[2, 0])),
                           'pitch': round(pitch, 1), 'yaw': round(yaw, 1), 'roll': round(roll, 1), 'accepted': True}
        if os.environ.get("DROWSY_DEBUG") and self._n % 15 == 0:
            print("[POSE]", self.last_debug)
        return {'pitch': pitch, 'yaw': yaw, 'roll': roll, 'rvec': rvec, 'tvec': tvec, 'success': True}

    @staticmethod
    def _rotation_matrix_to_euler(R):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy >= 1e-6:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0.0
        return math.degrees(x), math.degrees(y), math.degrees(z)

    def draw_axes(self, frame, rvec, tvec, length=50.0):
        axis_pts = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
        pts, _ = cv2.projectPoints(np.vstack([np.float32([[0, 0, 0]]), axis_pts]),
                                   rvec, tvec, self._camera_matrix, self._dist_coeffs)
        pts = pts.astype(int).reshape(-1, 2)
        o = tuple(pts[0])
        cv2.line(frame, o, tuple(pts[1]), (0, 0, 255), 3)
        cv2.line(frame, o, tuple(pts[2]), (0, 255, 0), 3)
        cv2.line(frame, o, tuple(pts[3]), (255, 0, 0), 3)

    def draw_pose_text(self, frame, pitch, yaw, roll, position=(10, 120)):
        x, y = position
        for i, (label, val) in enumerate([('Pitch', pitch), ('Yaw', yaw), ('Roll', roll)]):
            color = (0, 0, 255) if abs(val) > 20 else (0, 255, 0)
            cv2.putText(frame, f"{label}: {val:+.1f}", (x, y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def classify_head_direction(self, pitch, yaw, pitch_thresh=20.0, yaw_thresh=30.0):
        directions = []
        if pitch > pitch_thresh:
            directions.append('Down')
        elif pitch < -pitch_thresh:
            directions.append('Up')
        if yaw > yaw_thresh:
            directions.append('Right')
        elif yaw < -yaw_thresh:
            directions.append('Left')
        return 'Forward' if not directions else 'Looking ' + '/'.join(directions)