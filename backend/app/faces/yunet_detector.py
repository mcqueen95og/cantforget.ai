"""YuNet detector re-implemented on top of the ONNX base class.

This preserves the existing, working YuNet detection while conforming to the
new model-agnostic detector interface. Landmarks are extracted and returned
so they can be used for face alignment (Phase D).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .onnx_base import OnnxDetector

logger = logging.getLogger("forgetmenot.faces.yunet_detector")


class YuNetDetector(OnnxDetector):
    """OpenCV YuNet (FaceDetectorYN) wrapper via cv2.dnn."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.top_k = top_k
        self._legacy_input_size: Optional[tuple[int, int]] = None
        super().__init__(
            model_path=model_path,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            backend=backend,
            target=target,
        )
        # Create the legacy FaceDetectorYN API for landmark output.
        self._legacy = cv2.FaceDetectorYN_create(
            self.model_path,
            "",
            self.input_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
            backend,
            target,
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame.

        Returns a list of dicts with keys:
            x, y, w, h        (int pixels, top-left origin)
            confidence        (float 0..1)
            landmarks         (list of [x, y] or None)
            model             (str, detector model name)
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        # Set input size only when frame dimensions change.
        if self._legacy_input_size != (w, h):
            self._legacy.setInputSize((w, h))
            self._legacy_input_size = (w, h)

        _, faces = self._legacy.detect(frame)
        if faces is None:
            return []

        detections: list[dict] = []
        for face in faces:
            x, y, w_box, h_box = (
                int(face[0]),
                int(face[1]),
                int(face[2]),
                int(face[3]),
            )
            confidence = float(face[4]) if len(face) > 4 else 0.0
            landmarks: Optional[list[list[int]]] = None
            if len(face) >= 14:
                landmarks = [
                    [int(face[5]), int(face[6])],
                    [int(face[7]), int(face[8])],
                    [int(face[9]), int(face[10])],
                    [int(face[11]), int(face[12])],
                    [int(face[13]), int(face[14])] if len(face) > 14 else [0, 0],
                ]
            detections.append(
                {
                    "x": max(0, x),
                    "y": max(0, y),
                    "w": max(1, w_box),
                    "h": max(1, h_box),
                    "confidence": round(confidence, 4),
                    "landmarks": landmarks,
                    "model": "yunet",
                }
            )

        return detections


# Compatibility alias for existing imports
FaceDetector = YuNetDetector
