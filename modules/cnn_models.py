import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import cv2
import numpy as np

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]


def _points(landmarks, w, h):
    lm = getattr(landmarks, "landmark", landmarks)
    pts = np.array(
        [[p.x, p.y] if hasattr(p, "x") else list(p)[:2] for p in lm],
        dtype=float
    )
    if pts.max() <= 1.5:
        pts = pts * [w, h]
    return pts


class CNNModels:
    def __init__(self, base_dir):
        self.eye_model = None
        self.sg_model = None
        self._warned = False

        try:
            import keras

            model_dir = os.path.join(base_dir, "models")

            eye_path = os.path.join(model_dir, "eye_cnn_v2.keras")
            sg_path = os.path.join(model_dir, "sunglasses_cnn.keras")

            if os.path.exists(eye_path):
                self.eye_model = keras.models.load_model(
                    eye_path, compile=False
                )
                print("[INFO] Eye CNN loaded from models folder.")

            if os.path.exists(sg_path):
                self.sg_model = keras.models.load_model(
                    sg_path, compile=False
                )
                print("[INFO] Sunglasses CNN loaded from models folder.")

        except Exception as e:
            print(
                f"[WARN] CNN models not loaded, "
                f"using old methods: {e}"
            )

    @property
    def eye_ready(self):
        return self.eye_model is not None

    @property
    def sunglasses_ready(self):
        return self.sg_model is not None

    def _warn(self, e):
        if not self._warned:
            print(
                f"[WARN] CNN prediction failed "
                f"(check landmark format): {e}"
            )
            self._warned = True

    def sunglasses_prob(self, frame, landmarks):
        """Probability (0-1) that the driver wears sunglasses."""
        try:
            h, w = frame.shape[:2]
            pts = _points(landmarks, w, h)

            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)

            px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)

            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return None

            crop = cv2.cvtColor(
                cv2.resize(crop, (96, 96)),
                cv2.COLOR_BGR2RGB
            )

            p = self.sg_model.predict_on_batch(
                crop[None].astype("float32")
            )

            return float(np.asarray(p).reshape(-1)[0])

        except Exception as e:
            self._warn(e)
            return None

    def eye_open_prob(self, frame, landmarks):
        """Probability (0-1) that the eyes are open (average of both eyes)."""
        try:
            h, w = frame.shape[:2]
            pts = _points(landmarks, w, h)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crops = []

            for idx in (LEFT_EYE, RIGHT_EYE):
                p = pts[idx]

                cx, cy = p.mean(0)
                half = int(np.linalg.norm(p[0] - p[1]) * 0.9)

                x1 = int(max(0, cx - half))
                y1 = int(max(0, cy - half))
                x2 = int(min(w, cx + half))
                y2 = int(min(h, cy + half))

                c = gray[y1:y2, x1:x2]

                if c.shape[0] < 8 or c.shape[1] < 8:
                    continue

                c = cv2.equalizeHist(cv2.resize(c, (64, 64)))
                crops.append(c[..., None])

            if not crops:
                return None

            p = self.eye_model.predict_on_batch(
                np.stack(crops).astype("float32")
            )

            return float(np.mean(np.asarray(p)))

        except Exception as e:
            self._warn(e)
            return None
