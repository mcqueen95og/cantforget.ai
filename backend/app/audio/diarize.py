"""Speaker diarization using pretrained speaker embeddings (3D-Speaker ERes2Net).

Uses sherpa-onnx (ONNX Runtime) with the 3D-Speaker ERes2Net model to extract
512-dimensional speaker embeddings, then clusters them online with cosine
similarity. This replaces the earlier hand-made spectral fingerprint, which
could not reliably separate voices.

Assigns consistent temporary labels (SPEAKER_00, SPEAKER_01, ...). These are
NOT real-world identities -- they only distinguish voices within a session.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from ..config import SPEAKER_MODEL_PATH

logger = logging.getLogger("forgetmenot.audio.diarize")

SAMPLE_RATE = 16000

# The model is ~38 MB; load once and share it across sessions.
_extractor = None
_extractor_lock = threading.Lock()
_extractor_error: Optional[str] = None


def embed(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Return an L2-normalized 512-d speaker embedding, or None.

    Public helper shared by the diarizer, voice enrollment, and voice-to-face
    association so all stored and compared vectors live in the same space.
    """
    extractor = _get_extractor()
    if extractor is None:
        return None
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < int(sample_rate * 0.30):
        return None
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return None
    if peak > 0:
        audio = audio / peak
    try:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()
        vec = np.asarray(extractor.compute(stream), dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speaker embedding failed: %s", exc)
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return vec / norm


def _get_extractor():
    """Load and cache the shared speaker-embedding extractor."""
    global _extractor, _extractor_error
    if _extractor is not None:
        return _extractor
    with _extractor_lock:
        if _extractor is not None:
            return _extractor
        try:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=SPEAKER_MODEL_PATH,
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            _extractor = extractor
            _extractor_error = None
            logger.info("Speaker embedding model loaded: %s", SPEAKER_MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            _extractor_error = str(exc)
            logger.exception("Failed to load speaker embedding model: %s", exc)
            _extractor = None
        return _extractor


class SpeakerDiarizer:
    """Online diarization with pretrained ERes2Net speaker embeddings.

    Parameters
    ----------
    max_speakers:
        Maximum number of distinct speakers to track.
    similarity_threshold:
        Cosine similarity needed to join an existing speaker. For ERes2Net,
        same-speaker pairs are typically > 0.6 and different speakers < 0.3.
    min_segment_ms:
        Segments shorter than this reuse the previous speaker (too short for
        a reliable embedding).
    """

    def __init__(
        self,
        max_speakers: int = 4,
        similarity_threshold: float = 0.5,
        min_segment_ms: int = 500,
    ) -> None:
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.min_segment_ms = min_segment_ms

        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._next_index = 0
        self._last_speaker: Optional[str] = None
        # Model load happens on first use (inside a worker thread).
        self._load_attempted = False

    def reset(self) -> None:
        self._centroids.clear()
        self._counts.clear()
        self._next_index = 0
        self._last_speaker = None

    def _speaker_id(self) -> str:
        sid = f"SPEAKER_{self._next_index:02d}"
        self._next_index += 1
        return sid

    def is_loaded(self) -> bool:
        return _extractor is not None

    def _embed(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Return an L2-normalized speaker embedding, or None."""
        return embed(samples, SAMPLE_RATE)

    def _update(self, speaker_id: str, emb: np.ndarray) -> None:
        count = self._counts.get(speaker_id, 0)
        centroid = self._centroids.get(speaker_id)
        if centroid is None:
            self._centroids[speaker_id] = emb
        else:
            merged = (centroid * count + emb) / (count + 1)
            norm = float(np.linalg.norm(merged))
            self._centroids[speaker_id] = merged / norm if norm > 0 else merged
        self._counts[speaker_id] = count + 1
        self._last_speaker = speaker_id

    def assign(self, samples: np.ndarray, duration_ms: float) -> str:
        """Assign a temporary speaker label to one utterance."""
        # Too short for a reliable embedding: keep the previous speaker.
        if duration_ms < self.min_segment_ms:
            if self._last_speaker is not None:
                self._counts[self._last_speaker] = self._counts.get(self._last_speaker, 0) + 1
                return self._last_speaker
            if self._centroids:
                return max(self._counts.items(), key=lambda kv: kv[1])[0]
            sid = self._speaker_id()
            self._counts[sid] = 1
            self._last_speaker = sid
            return sid

        emb = self._embed(samples)
        if emb is None:
            if self._last_speaker is not None:
                return self._last_speaker
            if self._centroids:
                return max(self._counts.items(), key=lambda kv: kv[1])[0]
            sid = self._speaker_id()
            self._counts[sid] = 1
            self._last_speaker = sid
            return sid

        sims = {sid: float(np.dot(emb, c)) for sid, c in self._centroids.items()}
        best_sid = max(sims, key=sims.get) if sims else None
        best_sim = sims.get(best_sid, -1.0) if best_sid else -1.0

        # Hysteresis: if the previous speaker is nearly as close, stick with
        # them instead of flip-flopping between two similar centroids. Only
        # applies when the previous speaker is still a plausible match --
        # otherwise a genuinely new voice would never get its own label.
        if (
            self._last_speaker is not None
            and self._last_speaker in sims
            and sims[self._last_speaker] >= self.similarity_threshold - 0.10
            and sims[self._last_speaker] >= best_sim - 0.05
            and best_sim < self.similarity_threshold + 0.15
        ):
            chosen = self._last_speaker
        elif best_sid is not None and best_sim >= self.similarity_threshold:
            chosen = best_sid
        elif len(self._centroids) < self.max_speakers:
            chosen = self._speaker_id()
            logger.info(
                "Diarizer: new speaker %s (best cosine %.2f, speakers=%d)",
                chosen, best_sim, len(self._centroids) + 1,
            )
        elif best_sid is not None:
            chosen = best_sid
        else:
            chosen = self._speaker_id()

        self._update(chosen, emb)
        logger.debug(
            "Diarizer: %s (cosine=%.2f, threshold=%.2f, dur=%.0fms)",
            chosen, sims.get(chosen, best_sim), self.similarity_threshold, duration_ms,
        )
        return chosen
