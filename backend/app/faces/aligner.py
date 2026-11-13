"""Face alignment using facial landmarks.

Uses the 5 facial landmarks (left eye, right eye, nose, left mouth,
right mouth) detected by YuNet to compute a similarity transform that
aligns each face to a canonical reference pose. This produces consistent,
forward-facing face crops for recognition — fixing the core reliability
bug where unaligned, rotated crops produced inconsistent embeddings.

Reference landmarks are based on the standard ArcFace/InsightFace
112x112 alignment template.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("forgetmenot.faces.aligner")

# ArcFace/InsightFace reference landmarks for 112x112 aligned face.
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ],
    dtype=np.float32,
)


class FaceAligner:
    """Aligns face crops using detected facial landmarks.

    Computes a similarity transform (rotation + scale + translation)
    that maps detected landmarks to reference positions, then warps
    the face crop to a canonical 112x112 (or configurable) output.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (112, 112),
        reference_landmarks: Optional[np.ndarray] = None,
    ) -> None:
        self.output_size = output_size
        self.reference = (
            reference_landmarks
            if reference_landmarks is not None
            else REFERENCE_LANDMARKS_112
        )
        # Scale reference landmarks to output size if not 112x112.
        if output_size != (112, 112):
            scale_x = output_size[0] / 112.0
            scale_y = output_size[1] / 112.0
            self.reference = self.reference * np.array([scale_x, scale_y])

    def align(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        landmarks: Optional[list[list[int]]],
    ) -> np.ndarray:
        """Align a face from the full frame using landmarks.

        Parameters
        ----------
        frame:
            Full BGR frame.
        bbox:
            (x, y, w, h) face bounding box.
        landmarks:
            List of 5 [x, y] landmark points, or None.

        Returns
        -------
        np.ndarray
            Aligned BGR face crop of size `output_size`. Falls back to a
            simple resize of the raw crop if landmarks are unavailable
            or alignment fails.
        """
        x, y, w, h = bbox
        # Pad the crop slightly to avoid cutting off chin/ears.
        pad = 0.25
        x0 = max(0, int(x - w * pad))
        y0 = max(0, int(y - h * pad))
        x1 = min(frame.shape[1], int(x + w * (1 + pad)))
        y1 = min(frame.shape[0], int(y + h * (1 + pad)))
        crop = frame[y0:y1, x0:x1]

        if crop.size == 0:
            crop = frame[max(0, y):y + h, max(0, x):x + w]

        if landmarks is None or len(landmarks) < 5:
            logger.debug("No landmarks available; falling back to resize")
            return self._fallback_resize(crop)

        # Adjust landmarks to the padded crop coordinate system.
        lm = np.array(landmarks[:5], dtype=np.float32)
        lm[:, 0] -= x0
        lm[:, 1] -= y0

        try:
            # Estimate similarity transform from detected -> reference.
            transform, _ = cv2.estimateAffinePartial2D(
                lm.reshape(-1, 1, 2),
                self.reference.reshape(-1, 1, 2),
                method=cv2.LMEDS,
            )
            if transform is None:
                return self._fallback_resize(crop)

            aligned = cv2.warpAffine(
                crop,
                transform,
                self.output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned
        except cv2.error as exc:
            logger.warning("Face alignment failed: %s", exc)
            return self._fallback_resize(crop)

    def _fallback_resize(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)
        return cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)


def extract_landmarks_5(detection: dict) -> Optional[list[list[int]]]:
    """Extract the 5 primary landmarks from a detection dict.

    YuNet returns up to 5 landmarks. Returns None if unavailable.
    """
    lm = detection.get("landmarks")
    if lm is None or len(lm) < 5:
        return None
    return lm[:5]
