"""Voice Activity Detection using Silero VAD (PyTorch backend).

Lightweight CPU-friendly VAD. Uses the silero-vad package which downloads
the official Silero VAD model from PyTorch Hub on first use.
Classifies audio frames as speech or silence and aggregates contiguous
speech into segments with timestamps.

Independent from speech recognition -- VAD only decides *when* speech occurs,
not *what* is said.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("cantforget.audio.vad")

# Silero VAD expects 16 kHz mono audio.
SAMPLE_RATE = 16000


class SileroVAD:
    """Wrapper around Silero VAD (PyTorch model from silero-vad package).

    Parameters
    ----------
    threshold:
        Speech probability threshold (0..1). Higher = stricter.
    min_speech_ms:
        Minimum duration (ms) for a speech segment to be emitted.
    min_silence_ms:
        Minimum silence (ms) to split two speech segments.
    speech_pad_ms:
        Padding (ms) added to each side of a speech segment.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

        self._model: Optional[torch.nn.Module] = None
        self._utils: Optional[tuple] = None
        self._load()

    def _load(self) -> None:
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            from silero_vad import VADIterator, get_speech_timestamps
            self._utils = (VADIterator, get_speech_timestamps)
            logger.info("Silero VAD loaded (PyTorch model from silero-vad package)")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Silero VAD failed to load: %s", exc)
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def segment(self, samples: np.ndarray) -> list[dict]:
        """Convert raw audio into speech segments with timestamps.

        Parameters
        ----------
        samples:
            float32 numpy array of audio samples (1-D), 16 kHz mono.

        Returns
        -------
        list of dicts: {start_ms, end_ms, confidence, samples}.
        """
        if self._model is None:
            return []
        if samples.size == 0:
            return []

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < 64:
            return []

        # Normalize to [-1, 1].
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

        try:
            from silero_vad import get_speech_timestamps

            timestamps = get_speech_timestamps(
                audio,
                self._model,
                sampling_rate=SAMPLE_RATE,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=True,
            )

            segments: list[dict] = []
            for ts in timestamps:
                start_s = float(ts["start"])
                end_s = float(ts["end"])
                s0 = int(start_s * SAMPLE_RATE)
                s1 = int(end_s * SAMPLE_RATE)
                segments.append(
                    {
                        "start_ms": round(start_s * 1000.0, 1),
                        "end_ms": round(end_s * 1000.0, 1),
                        "confidence": 0.8,
                        "samples": audio[s0:s1].copy(),
                    }
                )
            return segments
        except Exception as exc:  # noqa: BLE001
            logger.exception("VAD segmentation failed: %s", exc)
            return []

    def reset(self) -> None:
        """Reset for a new audio stream (stateless, no-op)."""
        pass
