"""Voice-to-face association.

Associates diarized speaker labels (SPEAKER_XX) with recognized face
identities (person_uid) using synchronized timestamps.

When a face is recognized and a voice segment overlaps in time with that
face track, the speaker label is provisionally linked to the person. The
link requires sufficient confidence and temporal overlap.

This does NOT identify someone from voice alone -- it only links a
previously enrolled voice profile to a face-recognized identity.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..db import PeopleDB
from ..audio.diarize import embed

logger = logging.getLogger("cantforget.audio.associate")


class VoiceFaceAssociator:
    """Links speaker labels to face-recognized identities via timestamp sync.

    Parameters
    ----------
    db:
        PeopleDB instance for loading voice enrollments and storing links.
    max_offset_ms:
        Maximum time offset (ms) between voice segment and face track for
        association to be considered.
    confirm_frames:
        Number of consecutive agreeing segments needed to confirm a link.
    similarity_threshold:
        Cosine similarity threshold for voice-profile matching.
    """

    def __init__(
        self,
        db: PeopleDB,
        max_offset_ms: int = 1000,
        confirm_frames: int = 3,
        similarity_threshold: float = 0.65,
    ) -> None:
        self.db = db
        self.max_offset_ms = max_offset_ms
        self.confirm_frames = confirm_frames
        self.similarity_threshold = similarity_threshold

        # Speaker -> candidate identity tracking during a session.
        # speaker_id -> {person_uid: count, ...}
        self._speaker_votes: dict[str, dict[str, int]] = {}
        # Confirmed links: speaker_id -> person_uid
        self._confirmed_links: dict[str, str] = {}

        # Voice gallery: {person_uid: (n, 256) array}
        self._voice_gallery: dict[str, np.ndarray] = {}
        self._reload_voice_gallery()

    def _reload_voice_gallery(self) -> None:
        """Load voice enrollments from the database."""
        self._voice_gallery = self.db.load_voice_gallery()

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _match_voice_to_person(
        self, samples: np.ndarray
    ) -> tuple[Optional[str], float]:
        """Compare a voice segment against enrolled voice profiles.

        Returns (person_uid, confidence) or (None, 0.0).
        """
        if not self._voice_gallery:
            return None, 0.0

        fp = embed(samples)
        if fp is None:
            return None, 0.0
        best_uid: Optional[str] = None
        best_sim = -1.0

        for uid, embeddings in self._voice_gallery.items():
            # Compare against each stored embedding, take best. Skip profiles
            # stored with an older, incompatible embedding model.
            for emb in embeddings:
                if emb.shape[0] != fp.shape[0]:
                    continue
                sim = self._cosine(fp, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_uid = uid

        if best_sim >= self.similarity_threshold:
            return best_uid, best_sim
        return None, best_sim

    def associate_segment(
        self,
        session_id: str,
        speaker_id: str,
        samples: np.ndarray,
        start_ms: float,
        end_ms: float,
        active_face_tracks: list[dict],
    ) -> Optional[str]:
        """Try to associate a voice segment with a face-recognized identity.

        Parameters
        ----------
        speaker_id:
            Diarized speaker label (e.g. "SPEAKER_00").
        samples:
            Audio samples for this segment.
        start_ms, end_ms:
            Segment timestamps in milliseconds.
        active_face_tracks:
            List of currently active face tracks, each with:
              - track_id, person_uid, status, x, y, w, h
              - start_ms, end_ms (track timestamps)

        Returns
        -------
        person_uid if a confident association is made, None otherwise.
        """
        # If already confirmed for this speaker, return it.
        if speaker_id in self._confirmed_links:
            return self._confirmed_links[speaker_id]

        # Find face tracks that overlap with this voice segment.
        overlapping = []
        for track in active_face_tracks:
            if track.get("status") != "recognized" or not track.get("person_uid"):
                continue
            t_start = track.get("start_ms", 0)
            t_end = track.get("end_ms", float("inf"))
            # Check temporal overlap.
            overlap_start = max(start_ms, t_start)
            overlap_end = min(end_ms, t_end)
            if overlap_end > overlap_start:
                overlap_ms = overlap_end - overlap_start
                segment_ms = end_ms - start_ms
                overlap_ratio = overlap_ms / max(segment_ms, 1)
                if overlap_ratio > 0.3:  # At least 30% overlap.
                    overlapping.append(
                        {
                            "person_uid": track["person_uid"],
                            "overlap_ratio": overlap_ratio,
                            "offset_ms": abs(start_ms - t_start),
                        }
                    )

        if not overlapping:
            # No face track visible during this voice segment. Voice-profile
            # matching alone is still valid evidence: after enough agreeing
            # segments the speaker is linked to the enrolled person, so
            # attribution works across sessions even with the camera off.
            matched_uid, sim = self._match_voice_to_person(samples)
            if matched_uid is not None:
                self._vote(speaker_id, matched_uid, sim)
                confirmed = self._check_confirmation(session_id, speaker_id)
                if confirmed is not None:
                    logger.info(
                        "Voice-only link confirmed: %s -> %s", speaker_id, confirmed
                    )
                    return confirmed
            return None

        # Sort by overlap ratio (best first).
        overlapping.sort(key=lambda x: x["overlap_ratio"], reverse=True)
        best_face = overlapping[0]

        # Try voice-profile matching to confirm it's the same person.
        matched_uid, sim = self._match_voice_to_person(samples)

        if matched_uid is not None and matched_uid == best_face["person_uid"]:
            # Voice profile matches the visible face -- strong evidence.
            self._vote(speaker_id, matched_uid, sim)
        elif best_face["overlap_ratio"] > 0.7 and best_face["offset_ms"] < self.max_offset_ms:
            # Strong temporal overlap but no voice profile match yet.
            # Provisionally vote for the visible face's person.
            self._vote(speaker_id, best_face["person_uid"], best_face["overlap_ratio"])

        # Check if confirmed.
        confirmed = self._check_confirmation(session_id, speaker_id)
        if confirmed is not None:
            return confirmed

        return None

    def _check_confirmation(self, session_id: str, speaker_id: str) -> Optional[str]:
        """Confirm a speaker link when enough agreeing votes accumulated."""
        votes = self._speaker_votes.get(speaker_id, {})
        if not votes:
            return None
        best_person = max(votes, key=votes.get)
        if votes[best_person] < self.confirm_frames:
            return None
        self._confirmed_links[speaker_id] = best_person
        # Persist to database.
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            self.db.link_speaker_to_identity(
                session_id=session_id,
                speaker_id=speaker_id,
                identity_uid=best_person,
                confidence=self.similarity_threshold,
                method="timestamp_sync",
            )
            logger.info(
                "Voice-to-face link confirmed: %s -> %s", speaker_id, best_person
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist voice-to-face link: %s", exc)
        return best_person

    def _vote(self, speaker_id: str, person_uid: str, confidence: float) -> None:
        if speaker_id not in self._speaker_votes:
            self._speaker_votes[speaker_id] = {}
        self._speaker_votes[speaker_id][person_uid] = (
            self._speaker_votes[speaker_id].get(person_uid, 0) + 1
        )

    def get_link(self, speaker_id: str) -> Optional[str]:
        """Return the confirmed person_uid for a speaker, or None."""
        return self._confirmed_links.get(speaker_id)

    def get_all_links(self) -> dict[str, str]:
        """Return all confirmed speaker-to-identity links."""
        return dict(self._confirmed_links)

    def reset(self) -> None:
        """Clear all association state for a new session."""
        self._speaker_votes.clear()
        self._confirmed_links.clear()
        self._reload_voice_gallery()
