"""Face enrollment: capture samples, compute embeddings, store in SQLite.

Uses face alignment (Phase D) and model versioning (Phase E) to produce
consistent, high-quality embeddings. Validates sample quality before
accepting.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .aligner import FaceAligner
from .recognizer import FaceRecognizer
from .. import database

logger = logging.getLogger("cantforget.faces.enrollment")

# Minimum Laplacian variance for a non-blurry face sample.
MIN_BLUR_VARIANCE = 50.0
# Minimum face size (pixels) for enrollment.
MIN_FACE_SIZE = 60


def _blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher means sharper. Lower = blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class EnrollmentSession:
    """Collects face embeddings for a single person being enrolled.

    Validates each sample for blur, size, and detection confidence before
    accepting. Stores embeddings tagged with model version.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        aligner: FaceAligner,
        name: str,
        relationship: Optional[str] = None,
    ) -> None:
        self.recognizer = recognizer
        self.aligner = aligner
        self.name = name
        self.relationship = relationship
        self._embeddings: list[np.ndarray] = []
        self._embedding_meta: list[dict] = []
        self._person_id: Optional[str] = None

    def add_sample(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        landmarks: Optional[list[list[int]]] = None,
        detection_confidence: float = 0.0,
    ) -> dict:
        """Validate and add a face sample to the enrollment session.

        Parameters
        ----------
        frame:
            Full BGR frame (used for cropping).
        bbox:
            (x, y, w, h) face bounding box.
        landmarks:
            Optional 5 facial landmarks for alignment.
        detection_confidence:
            Detector confidence for this face.

        Returns
        -------
        dict
            {accepted: bool, reason: str, sample_count: int}
        """
        x, y, w, h = bbox

        # Validate face size
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return {
                "accepted": False,
                "reason": f"Face too small ({w}x{h}, min {MIN_FACE_SIZE})",
                "sample_count": len(self._embeddings),
            }

        # Validate detection confidence
        if detection_confidence < 0.6:
            return {
                "accepted": False,
                "reason": f"Low detection confidence ({detection_confidence:.2f})",
                "sample_count": len(self._embeddings),
            }

        # Align the face
        aligned = self.aligner.align(frame, bbox, landmarks)

        # Validate blur
        blur_var = _blur_score(aligned)
        if blur_var < MIN_BLUR_VARIANCE:
            return {
                "accepted": False,
                "reason": f"Image too blurry (variance {blur_var:.1f}, min {MIN_BLUR_VARIANCE})",
                "sample_count": len(self._embeddings),
            }

        # Compute embedding
        emb = self.recognizer.embed(aligned)
        if emb is None:
            return {
                "accepted": False,
                "reason": "Embedding computation failed",
                "sample_count": len(self._embeddings),
            }

        self._embeddings.append(emb)
        self._embedding_meta.append(
            {
                "blur_variance": round(blur_var, 2),
                "detection_confidence": round(detection_confidence, 4),
                "face_size": [w, h],
            }
        )
        logger.debug(
            "Enrollment sample %d accepted for %s (blur=%.1f, conf=%.3f)",
            len(self._embeddings),
            self.name,
            blur_var,
            detection_confidence,
        )
        return {
            "accepted": True,
            "reason": "ok",
            "sample_count": len(self._embeddings),
            "blur_variance": round(blur_var, 2),
        }

    @property
    def sample_count(self) -> int:
        return len(self._embeddings)

    @property
    def person_id(self) -> Optional[str]:
        return self._person_id

    def commit(self, min_samples: int = 3) -> tuple[bool, str]:
        """Persist the enrolled person and embeddings to the database.

        Tags each embedding with the current model version.
        """
        if len(self._embeddings) < min_samples:
            return False, (
                f"Need at least {min_samples} valid samples, have {len(self._embeddings)}"
            )

        # Create person record
        person = database.create_person(self.name, self.relationship)
        self._person_id = person["id"]

        # Store each embedding with model version
        model_version = self.recognizer.model_version
        for emb in self._embeddings:
            emb_bytes = emb.astype(np.float32).tobytes()
            database.add_embedding_with_version(self._person_id, emb_bytes, model_version)

        logger.info(
            "Enrolled %s (id=%s) with %d embeddings (model=%s)",
            self.name,
            self._person_id,
            len(self._embeddings),
            model_version,
        )
        return True, f"Enrolled {self.name} with {len(self._embeddings)} samples"

    def reset(self) -> None:
        self._embeddings.clear()
        self._embedding_meta.clear()
        self._person_id = None
