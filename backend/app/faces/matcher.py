"""Embedding comparison with a conservative "never guess" policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MATCH = "match"
UNCERTAIN = "uncertain"
UNKNOWN = "unknown"


def normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(frozen=True)
class MatchResult:
    status: str  # MATCH | UNCERTAIN | UNKNOWN
    person_uid: Optional[str] = None  # set only for MATCH
    score: float = 0.0  # best cosine similarity seen (diagnostic)
    best_uid: Optional[str] = None  # best candidate even if not accepted (internal)


class Matcher:
    """Cosine-similarity matcher over an in-memory gallery.

    * MATCH     best >= threshold and beats the runner-up *person* by margin.
    * UNCERTAIN best in [unknown_below, threshold), or too close to another person.
    * UNKNOWN   best < unknown_below, or the gallery is empty.
    """

    def __init__(self, threshold: float = 0.45, margin: float = 0.05, unknown_below: float = 0.30) -> None:
        if not (0.0 <= unknown_below <= threshold <= 1.0):
            raise ValueError("Require 0 <= unknown_below <= threshold <= 1")
        self.threshold = threshold
        self.margin = margin
        self.unknown_below = unknown_below
        self._gallery: dict[str, np.ndarray] = {}

    def set_gallery(self, gallery: dict[str, np.ndarray]) -> None:
        self._gallery = {
            uid: np.vstack([normalize(row) for row in np.atleast_2d(arr)])
            for uid, arr in gallery.items()
            if len(np.atleast_2d(arr)) > 0
        }

    @property
    def size(self) -> int:
        return len(self._gallery)

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        q = normalize(embedding)
        return {uid: float(np.max(arr @ q)) for uid, arr in self._gallery.items()}

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._gallery:
            return MatchResult(UNKNOWN)
        ranked = sorted(self.scores(embedding).items(), key=lambda kv: kv[1], reverse=True)
        best_uid, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        best = round(best, 4)
        if best < self.unknown_below:
            return MatchResult(UNKNOWN, None, best, best_uid)
        if best >= self.threshold and (best - second) >= self.margin:
            return MatchResult(MATCH, best_uid, best, best_uid)
        return MatchResult(UNCERTAIN, None, best, best_uid)
