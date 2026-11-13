"""Identity service: ties detection output to recognition, registration and
new-face candidates. Pure Python + numpy + OpenCV, no web framework, so it is
unit-testable with a fake recognizer.

Policy (from the product brief):
  * Only enrolled, consented people are ever named.
  * A weak or ambiguous match is "uncertain" -- never a guessed name, and
    never auto-offered for registration (it might be an enrolled person).
  * A genuinely unknown face becomes an in-memory *candidate*; it is stored
    only if a caregiver confirms it with explicit consent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..db import PeopleDB, ValidationError, clean_profile
from .matcher import MATCH, UNCERTAIN, UNKNOWN, Matcher, MatchResult
from .pending import PendingStore
from .recognizer import make_thumbnail

logger = logging.getLogger("forgetmenot.faces.identity")


class DuplicatePersonError(Exception):
    """The face looks like an already-registered person."""

    def __init__(self, person: dict) -> None:
        super().__init__(f"This face looks like {person['name']}, who is already registered.")
        self.person = person


class NoUsableFaceError(ValueError):
    """No single, clear face could be found in the supplied image(s)."""


@dataclass
class _TrackState:
    status: str = "checking"
    uid: Optional[str] = None
    score: float = 0.0
    confirm_uid: Optional[str] = None
    confirm_count: int = 0
    bad: int = 0
    last_check: int = -10**9
    last_seen: int = 0
    unknown_embs: list = field(default_factory=list)
    thumb: Optional[bytes] = None
    thumb_score: float = 0.0
    pending_id: Optional[str] = None


class IdentityService:
    def __init__(
        self,
        recognizer: Any,
        db: PeopleDB,
        matcher: Matcher,
        pending: PendingStore,
        detector: Any = None,
        *,
        confirm_frames: int = 2,
        reverify_frames: int = 10,
        min_face_px: int = 70,
        min_face_confidence: float = 0.8,
        pending_min_samples: int = 4,
        pending_max_samples: int = 5,
    ) -> None:
        self.recognizer = recognizer
        self.db = db
        self.matcher = matcher
        self.pending = pending
        self.detector = detector
        self.confirm_frames = confirm_frames
        self.reverify_frames = reverify_frames
        self.min_face_px = min_face_px
        self.min_face_confidence = min_face_confidence
        self.pending_min_samples = pending_min_samples
        self.pending_max_samples = pending_max_samples
        self._tracks: dict[int, _TrackState] = {}
        self._people: dict[str, dict] = {}
        self._frame = 0
        self.reload()

    # ------------------------------------------------------------------ state

    def reload(self) -> None:
        """Refresh the matching gallery and profile cache from the database."""
        self.matcher.set_gallery(self.db.load_gallery())
        self._people = {p["uid"]: p for p in self.db.list_people()}

    def _reset_tracks(self, *, uid: Optional[str] = None, pending_id: Optional[str] = None) -> None:
        for tid, st in list(self._tracks.items()):
            if (uid and st.uid == uid) or (pending_id and st.pending_id == pending_id):
                self._tracks[tid] = _TrackState(last_seen=self._frame)

    # --------------------------------------------------------------- per frame

    def process(self, frame: np.ndarray, faces: list[dict]) -> dict[int, dict]:
        """Attach an identity result to each tracked face.

        ``faces``: dicts with track_id, x, y, w, h, confidence, row.
        Returns {track_id: identity_dict} (safe to serialise to the client).
        """
        self._frame += 1
        self.pending.purge()
        out: dict[int, dict] = {}
        for f in faces:
            st = self._tracks.setdefault(f["track_id"], _TrackState())
            st.last_seen = self._frame
            eligible = (
                min(f["w"], f["h"]) >= self.min_face_px and f["confidence"] >= self.min_face_confidence
            )
            settled = st.status in ("recognized", "new_face")
            due = (not settled) or (self._frame - st.last_check) >= self.reverify_frames
            if eligible and due:
                try:
                    emb = self.recognizer.embed(frame, f["row"])
                except Exception:  # noqa: BLE001 - one bad face must not break the frame
                    logger.exception("Embedding failed for track %s", f["track_id"])
                else:
                    st.last_check = self._frame
                    self._apply(st, self.matcher.match(emb), emb, frame, f)
            elif not eligible and st.status not in ("recognized", "new_face"):
                st.status = "low_quality"
            out[f["track_id"]] = self._public(st)

        # Forget states of tracks that have been gone for a while.
        for tid in [t for t, s in self._tracks.items() if self._frame - s.last_seen > 100]:
            del self._tracks[tid]
        return out

    def _apply(self, st: _TrackState, res: MatchResult, emb: np.ndarray, frame: np.ndarray, f: dict) -> None:
        if res.status == MATCH:
            st.unknown_embs.clear()
            st.pending_id = None
            if st.confirm_uid == res.person_uid:
                st.confirm_count += 1
            else:
                st.confirm_uid, st.confirm_count = res.person_uid, 1
            if st.status == "recognized" and st.uid == res.person_uid:
                st.score, st.bad = res.score, 0
            elif st.confirm_count >= self.confirm_frames:
                st.status, st.uid, st.score, st.bad = "recognized", res.person_uid, res.score, 0
            elif st.status != "recognized":
                st.status = "checking"
            return

        st.confirm_uid, st.confirm_count = None, 0

        if st.status == "recognized":
            # Tolerate a momentary bad frame; drop the name after repeated misses.
            st.bad += 1
            if st.bad < 3:
                return
            st.status, st.uid, st.score, st.bad = (
                "uncertain" if res.status == UNCERTAIN else "unknown",
                None,
                res.score,
                0,
            )
            return

        if res.status == UNCERTAIN:
            st.status, st.score = "uncertain", res.score
            st.unknown_embs.clear()
            return

        # UNKNOWN: collect samples toward a registration candidate.
        st.status, st.score = "unknown", res.score
        if len(st.unknown_embs) < self.pending_max_samples:
            st.unknown_embs.append(emb)
        q = f["w"] * f["h"] * f["confidence"]
        if st.thumb is None or q > st.thumb_score:
            thumb = make_thumbnail(frame, f["x"], f["y"], f["w"], f["h"])
            if thumb is not None:
                st.thumb, st.thumb_score = thumb, q
        if len(st.unknown_embs) >= self.pending_min_samples:
            cand = self.pending.upsert(st.unknown_embs, st.thumb, st.thumb_score)
            st.pending_id = cand.id
            st.status = "unknown" if cand.dismissed else "new_face"

    def _public(self, st: _TrackState) -> dict:
        if st.status == "recognized" and st.uid in self._people:
            p = self._people[st.uid]
            return {
                "status": "recognized",
                "person_uid": p["uid"],
                "name": p["name"],
                "tag": p["tag"],
                "description": p["description"],
                "score": round(st.score, 3),
            }
        if st.status == "new_face" and st.pending_id:
            self.pending.touch(st.pending_id)
            return {"status": "new_face", "pending_id": st.pending_id}
        if st.status == "recognized":  # profile vanished (deleted) -> not a name
            return {"status": "checking"}
        return {"status": st.status}

    # ------------------------------------------------------------ registration

    def _check_duplicate(self, embeddings: list[np.ndarray]) -> None:
        for e in embeddings:
            res = self.matcher.match(e)
            if res.status == MATCH and res.person_uid in self._people:
                raise DuplicatePersonError(self._people[res.person_uid])

    def register_pending(
        self, pending_id: str, name: str, tag: str, description: str, consent: bool
    ) -> dict:
        if not consent:
            raise ValidationError("Consent is required before registering a face.")
        clean_profile(name, tag, description)  # validate before touching anything
        cand = self.pending.get(pending_id)
        if cand is None:
            raise KeyError(pending_id)
        self._check_duplicate(cand.embeddings)
        person = self.db.create_person(
            name, tag, description, cand.embeddings, cand.thumbnail, consent=True
        )
        self.pending.remove(pending_id)
        self._reset_tracks(pending_id=pending_id)
        self.reload()
        return person

    def dismiss_pending(self, pending_id: str) -> bool:
        ok = self.pending.dismiss(pending_id)
        if ok:
            self._reset_tracks(pending_id=pending_id)
        return ok

    def _embed_single_face(self, image: np.ndarray) -> tuple[np.ndarray, Optional[bytes], float]:
        if self.detector is None:
            raise NoUsableFaceError("Face detector is not available.")
        dets = [d for d in self.detector.detect(image) if d["confidence"] >= self.min_face_confidence - 0.2]
        if not dets:
            raise NoUsableFaceError("No face found in the image.")
        dets.sort(key=lambda d: d["w"] * d["h"], reverse=True)
        best = dets[0]
        if len(dets) > 1 and dets[1]["w"] * dets[1]["h"] >= 0.5 * best["w"] * best["h"]:
            raise NoUsableFaceError("More than one face in the image; use a photo of one person.")
        if min(best["w"], best["h"]) < 48:
            raise NoUsableFaceError("The face is too small; use a closer photo.")
        emb = self.recognizer.embed(image, best["row"])
        thumb = make_thumbnail(image, best["x"], best["y"], best["w"], best["h"])
        return emb, thumb, best["w"] * best["h"] * best["confidence"]

    def _embed_images(self, images: list[np.ndarray]) -> tuple[list[np.ndarray], Optional[bytes], list[str]]:
        embs: list[np.ndarray] = []
        notes: list[str] = []
        thumb, thumb_q = None, -1.0
        for i, img in enumerate(images, 1):
            try:
                emb, t, q = self._embed_single_face(img)
            except NoUsableFaceError as exc:
                notes.append(f"Photo {i}: {exc}")
                continue
            embs.append(emb)
            if t is not None and q > thumb_q:
                thumb, thumb_q = t, q
        return embs, thumb, notes

    def enroll_from_images(
        self, name: str, tag: str, description: str, consent: bool, images: list[np.ndarray]
    ) -> tuple[dict, list[str]]:
        if not consent:
            raise ValidationError("Consent is required before registering a face.")
        clean_profile(name, tag, description)
        embs, thumb, notes = self._embed_images(images)
        if not embs:
            raise NoUsableFaceError(" ".join(notes) or "No usable face found.")
        self._check_duplicate(embs)
        person = self.db.create_person(name, tag, description, embs, thumb, consent=True)
        self.reload()
        return person, notes

    def add_samples(self, uid: str, images: list[np.ndarray]) -> tuple[dict, list[str]]:
        if uid not in self._people:
            raise KeyError(uid)
        embs, _, notes = self._embed_images(images)
        if not embs:
            raise NoUsableFaceError(" ".join(notes) or "No usable face found.")
        # Safety: new samples must not look like a *different* enrolled person.
        for e in embs:
            res = self.matcher.match(e)
            if res.status == MATCH and res.person_uid != uid:
                raise DuplicatePersonError(self._people[res.person_uid])
        self.db.add_embeddings(uid, embs)
        self.reload()
        return self._people[uid], notes

    def update_person(self, uid: str, name: str, tag: str, description: str) -> dict:
        person = self.db.update_person(uid, name, tag, description)
        self.reload()
        return person

    def delete_person(self, uid: str) -> bool:
        ok = self.db.delete_person(uid)
        if ok:
            self._reset_tracks(uid=uid)
            self.reload()
        return ok
