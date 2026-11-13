"""Face embeddings using OpenCV SFace (FaceRecognizerSF).

CPU-only. Takes the raw YuNet detection row (box + 5 landmarks) so SFace can
align the face before computing a 128-d embedding.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .matcher import normalize

logger = logging.getLogger("forgetmenot.faces.recognizer")


class FaceRecognizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SFace model not found: {self.model_path}")
        create = getattr(cv2.FaceRecognizerSF, "create", None) or cv2.FaceRecognizerSF_create
        self._rec = create(self.model_path, "")
        self._lock = threading.Lock()
        logger.info("SFace recognizer loaded: model=%s", self.model_path)

    def embed(self, frame: np.ndarray, detection_row: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 128-d embedding for one YuNet detection row."""
        row = np.asarray(detection_row, dtype=np.float32).reshape(-1)
        with self._lock:
            aligned = self._rec.alignCrop(frame, row)
            feat = self._rec.feature(aligned)
        return normalize(np.asarray(feat, dtype=np.float32).reshape(-1))


def make_thumbnail(
    frame: np.ndarray, x: int, y: int, w: int, h: int, size: int = 128, margin: float = 0.25
) -> Optional[bytes]:
    """Square JPEG crop around a face (with margin) for the caregiver to review."""
    fh, fw = frame.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * (1.0 + margin) / 2.0
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(fw, cx + half)), int(min(fh, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None
