"""Base class for ONNX Runtime face detectors.

Provides a common interface for any ONNX-based face detector (YuNet, SCRFD,
MediaPipe, etc.) so the rest of the pipeline is model-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("forgetmenot.faces.onnx_base")


class OnnxDetector:
    """Base wrapper for OpenCV DNN-based ONNX face detectors."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.model_path = str(model_path)
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.backend = backend
        self.target = target

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Detector model not found: {self.model_path}")

        self._net: Optional[cv2.dnn.Net] = None
        self._load()

    def _load(self) -> None:
        try:
            self._net = cv2.dnn.readNetFromONNX(self.model_path)
            self._net.setPreferableBackend(self.backend)
            self._net.setPreferableTarget(self.target)
            logger.info(
                "ONNX detector loaded: model=%s input_size=%s backend=%s target=%s",
                self.model_path,
                self.input_size,
                self.backend,
                self.target,
            )
        except cv2.error as exc:
            logger.exception("Failed to load ONNX detector %s", self.model_path)
            raise

    @property
    def net(self) -> cv2.dnn.Net:
        if self._net is None:
            raise RuntimeError("Detector network not loaded")
        return self._net

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Subclasses must implement detection logic."""
        raise NotImplementedError
