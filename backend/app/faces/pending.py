"""In-memory store of *unregistered* faces awaiting a caregiver's decision.

Nothing here is written to disk. A new face seen by the camera becomes a
"pending" candidate (a few embeddings + one thumbnail) that expires after a
TTL. It only becomes an enrolled person when a human confirms with consent.
Dismissed candidates are remembered (in memory, until TTL) so the same face
is not offered again and again.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .matcher import normalize


@dataclass
class Candidate:
    id: str
    embeddings: list[np.ndarray]
    thumbnail: Optional[bytes]
    thumb_score: float
    created: float
    last_seen: float
    dismissed: bool = False
    sightings: int = 1


class PendingStore:
    def __init__(
        self,
        same_person_threshold: float,
        ttl_seconds: float = 900,
        max_candidates: int = 8,
        max_samples: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.same_person_threshold = same_person_threshold
        self.ttl = ttl_seconds
        self.max_candidates = max_candidates
        self.max_samples = max_samples
        self._clock = clock
        self._items: dict[str, Candidate] = {}

    def purge(self) -> None:
        now = self._clock()
        for cid in [c.id for c in self._items.values() if now - c.last_seen > self.ttl]:
            del self._items[cid]

    def _same(self, embs: list[np.ndarray], cand: Candidate) -> bool:
        a = np.vstack([normalize(e) for e in embs])
        b = np.vstack([normalize(e) for e in cand.embeddings])
        # Average best-match similarity: robust to one odd sample.
        return float(np.mean(np.max(a @ b.T, axis=1))) >= self.same_person_threshold

    def upsert(
        self, embeddings: list[np.ndarray], thumbnail: Optional[bytes], thumb_score: float
    ) -> Candidate:
        """Merge into a matching candidate or create a new one."""
        self.purge()
        now = self._clock()
        for cand in self._items.values():
            if self._same(embeddings, cand):
                cand.last_seen = now
                cand.sightings += 1
                if len(cand.embeddings) < self.max_samples:
                    cand.embeddings.extend(embeddings[: self.max_samples - len(cand.embeddings)])
                if thumbnail is not None and thumb_score > cand.thumb_score:
                    cand.thumbnail, cand.thumb_score = thumbnail, thumb_score
                return cand
        if len(self._items) >= self.max_candidates:
            oldest = min(self._items.values(), key=lambda c: c.last_seen)
            del self._items[oldest.id]
        cand = Candidate(
            id="n_" + uuid.uuid4().hex[:8],
            embeddings=list(embeddings[: self.max_samples]),
            thumbnail=thumbnail,
            thumb_score=thumb_score,
            created=now,
            last_seen=now,
        )
        self._items[cand.id] = cand
        return cand

    def get(self, cid: str) -> Optional[Candidate]:
        self.purge()
        return self._items.get(cid)

    def touch(self, cid: str) -> None:
        c = self._items.get(cid)
        if c:
            c.last_seen = self._clock()

    def dismiss(self, cid: str) -> bool:
        c = self.get(cid)
        if c is None:
            return False
        c.dismissed = True
        c.last_seen = self._clock()
        return True

    def remove(self, cid: str) -> bool:
        return self._items.pop(cid, None) is not None

    def list_active(self) -> list[Candidate]:
        self.purge()
        return [c for c in self._items.values() if not c.dismissed]
