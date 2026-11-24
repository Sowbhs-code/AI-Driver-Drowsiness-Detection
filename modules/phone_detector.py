"""
phone_detector.py
─────────────────────────────────────────────────────────
Mobile phone detection using YOLOv8 (Ultralytics).

COCO class index for 'cell phone' = 67
The detector loads yolov8n.pt (nano) by default; swap in
a fine-tuned model path for better accuracy.
─────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import os

# Graceful fallback if ultralytics is not yet installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – PhoneDetector running in mock mode.")


class PhoneDetector:
    """
    Detects mobile phones in a BGR frame using YOLOv8.

    Parameters
    ----------
    model_path       : str    path to .pt weights file
    conf_threshold   : float  minimum confidence to report a detection
    iou_threshold    : float  NMS IoU threshold
    device           : str    'cpu' | 'cuda' | 'mps'
    """

    PHONE_CLASS_ID = 67      # COCO class index for 'cell phone'
    PHONE_CLASS_NAMES = {'cell phone', 'mobile phone', 'phone'}

    def __init__(self, model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.50,
                 iou_threshold: float = 0.45,
                 device: str = 'cpu'):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device
        self.model          = None
        self._load_model(model_path)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self, model_path: str) -> None:
        if not YOLO_AVAILABLE:
            return
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Auto-download YOLOv8n weights from Ultralytics hub
                print(f"[INFO] '{model_path}' not found – downloading yolov8n.pt...")
                self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            print(f"[INFO] YOLOv8 model loaded on {self.device}.")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            self.model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run phone detection on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image

        Returns
        -------
        dict
            phone_found : bool
            boxes       : list of (x1, y1, x2, y2) int tuples
            confidences : list of float
            count       : int
        """
        if self.model is None:
            return self._mock_result()

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}")
            return self._mock_result()

        boxes       = []
        confidences = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                # Check COCO class ID or class name
                cls_name = self.model.names.get(cls_id, '').lower()
                if cls_id == self.PHONE_CLASS_ID or cls_name in self.PHONE_CLASS_NAMES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(conf)

        return {
            'phone_found': len(boxes) > 0,
            'boxes': boxes,
            'confidences': confidences,
            'count': len(boxes),
        }

    # ── Drawing Utilities ─────────────────────────────────────────────────────

    def draw_detections(self, frame: np.ndarray, detection_result: dict) -> None:
        """Overlay bounding boxes and labels for detected phones."""
        for (x1, y1, x2, y2), conf in zip(
                detection_result.get('boxes', []),
                detection_result.get('confidences', [])):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Phone {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    _warned = False

    @classmethod
    def _mock_result(cls) -> dict:
        """Model unavailable: report it loudly (once) and flag the result with an 'error' key."""
        if not cls._warned:
            print("[ERROR] PhoneDetector has no model loaded - phone detection is NOT running.")
            cls._warned = True
        return {'phone_found': False, 'boxes': [], 'confidences': [], 'count': 0,
                'error': 'phone model not loaded'}

    def set_confidence_threshold(self, threshold: float) -> None:
        self.conf_threshold = max(0.1, min(1.0, threshold))

    def get_model_info(self) -> dict:
        if self.model is None:
            return {'available': False}
        return {
            'available': True,
            'model_type': type(self.model).__name__,
            'num_classes': len(self.model.names) if hasattr(self.model, 'names') else 0,
            'device': self.device,
        }
