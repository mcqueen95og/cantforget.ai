"""Face detection using OpenCV YuNet (FaceDetectorYN).

CPU-friendly, no GPU required. Returns bounding boxes with confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("forgetmenot.faces.detector")


class FaceDetector:
    """Wrapper around OpenCV's YuNet face detector.

    The detector is stateless per call; callers are responsible for
    supplying frames and reading back detections.
    """

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
        self.model_path = str(model_path)
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YuNet model not found: {self.model_path}")

        try:
            self._detector = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                self.input_size,
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
                backend,
                target,
            )
            logger.info(
                "YuNet detector loaded: model=%s input_size=%s",
                self.model_path,
                self.input_size,
            )
        except cv2.error as exc:
            logger.exception("Failed to load YuNet model %s", self.model_path)
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame.

        YuNet returns one 15-value row per face:
            [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
             rmouth_x, rmouth_y, lmouth_x, lmouth_y, score]
        i.e. the 5 landmarks are indices 4..13 and the score is index 14.

        Returns a list of dicts with keys:
            x, y, w, h  (int pixels, top-left origin)
            confidence  (float 0..1)
            landmarks   (5 x [x, y] ints)
            row         (the raw float32 YuNet row; needed by SFace for
                         alignment. Internal -- never serialise to clients.)
        """
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]
        # YuNet requires the input size to match the frame being processed.
        self._detector.setInputSize((fw, fh))

        _, faces = self._detector.detect(frame)
        if faces is None:
            return []

        detections: list[dict] = []
        for face in faces:
            if len(face) < 15:
                continue
            x, y, w_box, h_box = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
            landmarks = [[int(face[4 + 2 * i]), int(face[5 + 2 * i])] for i in range(5)]
            detections.append(
                {
                    "x": max(0, x),
                    "y": max(0, y),
                    "w": max(1, w_box),
                    "h": max(1, h_box),
                    "confidence": round(float(face[14]), 4),
                    "landmarks": landmarks,
                    "row": np.asarray(face, dtype=np.float32).copy(),
                }
            )

        return detections
