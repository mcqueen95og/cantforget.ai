"""Voice enrollment: explicit consent-based voice profile creation.

Participants opt into voice enrollment. The application captures a short
calibration recording, generates a speaker embedding, and associates it
with their existing authorized identity (from face recognition).

Voice profiles are stored separately from face embeddings and require
explicit consent. They can be deleted independently.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..db import PeopleDB
from ..audio.vad import SileroVAD
from ..audio.diarize import embed

logger = logging.getLogger("forgetmenot.audio.enrollment")


class VoiceEnrollmentSession:
    """Collects voice samples for a single person during enrollment.

    Not thread-safe; intended for one enrollment at a time.
    """

    def __init__(
        self,
        person_uid: str,
        name: str,
        min_samples: int = 3,
        max_samples: int = 10,
    ) -> None:
        self.person_uid = person_uid
        self.name = name
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._fingerprints: list[np.ndarray] = []
        self._sample_durations: list[float] = []

    def add_sample(
        self,
        samples: np.ndarray,
        duration_ms: float,
    ) -> dict:
        """Validate and add a voice sample.

        Parameters
        ----------
        samples:
            Audio samples (16 kHz mono float32).
        duration_ms:
            Duration of the sample in milliseconds.

        Returns
        -------
        dict with {accepted: bool, reason: str, sample_count: int}
        """
        if samples.size == 0:
            return {"accepted": False, "reason": "Empty audio sample", "sample_count": len(self._fingerprints)}

        if duration_ms < 1000:
            return {
                "accepted": False,
                "reason": f"Sample too short ({duration_ms:.0f} ms, min 1000 ms)",
                "sample_count": len(self._fingerprints),
            }

        if len(self._fingerprints) >= self.max_samples:
            return {
                "accepted": False,
                "reason": f"Maximum {self.max_samples} samples reached",
                "sample_count": len(self._fingerprints),
            }

        fp = embed(samples)
        if fp is None:
            return {
                "accepted": False,
                "reason": "Could not extract a stable voice embedding from this sample",
                "sample_count": len(self._fingerprints),
            }
        self._fingerprints.append(fp)
        self._sample_durations.append(duration_ms)

        logger.debug(
            "Voice enrollment sample %d accepted for %s (%.0f ms)",
            len(self._fingerprints),
            self.name,
            duration_ms,
        )
        return {
            "accepted": True,
            "reason": "ok",
            "sample_count": len(self._fingerprints),
        }

    @property
    def sample_count(self) -> int:
        return len(self._fingerprints)

    def compute_embedding(self) -> Optional[np.ndarray]:
        """Compute the average voice fingerprint from collected samples."""
        if not self._fingerprints:
            return None
        emb = np.mean(np.vstack(self._fingerprints), axis=0)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def commit(self, db: PeopleDB, consent_at: str) -> tuple[bool, str]:
        """Persist the enrolled voice profile to the database."""
        if self.sample_count < self.min_samples:
            return False, (
                f"Need at least {self.min_samples} voice samples, have {self.sample_count}"
            )

        embedding = self.compute_embedding()
        if embedding is None:
            return False, "Failed to compute voice embedding"

        try:
            eid = db.add_voice_enrollment(
                person_uid=self.person_uid,
                embedding=embedding,
                consent_at=consent_at,
                model_version="eres2net:512d",
            )
            logger.info(
                "Voice enrolled %s (id=%s) with %d samples (eid=%s)",
                self.name,
                self.person_uid,
                self.sample_count,
                eid,
            )
            return True, f"Voice enrolled {self.name} with {self.sample_count} samples"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to store voice enrollment")
            return False, f"Database error: {exc}"

    def reset(self) -> None:
        self._fingerprints.clear()
        self._sample_durations.clear()


# In-memory enrollment sessions: enrollment_id -> VoiceEnrollmentSession
_voice_enrollment_sessions: dict[str, VoiceEnrollmentSession] = {}


def start_voice_enrollment(
    person_uid: str,
    name: str,
    min_samples: int = 3,
    max_samples: int = 10,
) -> str:
    """Start a new voice enrollment session. Returns enrollment_id."""
    eid = "ve_" + str(__import__("uuid").uuid4().hex[:12])
    session = VoiceEnrollmentSession(person_uid, name, min_samples, max_samples)
    _voice_enrollment_sessions[eid] = session
    return eid


def get_voice_enrollment(enrollment_id: str) -> Optional[VoiceEnrollmentSession]:
    return _voice_enrollment_sessions.get(enrollment_id)


def delete_voice_enrollment_session(enrollment_id: str) -> bool:
    if enrollment_id in _voice_enrollment_sessions:
        del _voice_enrollment_sessions[enrollment_id]
        return True
    return False
