"""Speech-to-Text using faster-whisper (CTranslate2).

CPU-optimized Whisper implementation. Uses the "tiny" model by default for
low RAM usage and fast inference on an i3-class CPU.

Transcribes 16 kHz mono audio segments into text with timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("cantforget.audio.asr")

# Whisper expects 16 kHz mono audio.
SAMPLE_RATE = 16000


class WhisperASR:
    """Wrapper around faster-whisper.

    Parameters
    ----------
    model_size:
        Whisper model size: "tiny", "base", "small", "medium", "large-v3".
        "tiny" recommended for i3/8GB CPU-only hardware.
    device:
        "cpu" or "cuda".
    compute_type:
        "int8" (fastest, lowest RAM), "float16", or "float32".
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None
        self._load(cpu_threads)

    def _load(self, cpu_threads: int = 4) -> None:
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=max(1, cpu_threads),
            )
            logger.info(
                "Whisper ASR loaded: model=%s device=%s compute=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Whisper ASR failed to load: %s", exc)
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe_segments(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
    ) -> list[dict]:
        """Transcribe and return each Whisper sentence separately.

        Each item: {text, start_ms, end_ms, language, language_probability}.
        Splitting per sentence lets each utterance get its own speaker label.
        """
        if self._model is None or samples.size == 0:
            return []

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,  # Beam search: noticeably better accuracy on CPU
                vad_filter=False,  # We already do VAD separately
                condition_on_previous_text=False,  # Independent segments
            )
            lang = info.language if info else None
            lang_prob = (
                round(info.language_probability, 3)
                if info and info.language_probability
                else None
            )
            out: list[dict] = []
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                # Drop low-quality output: hallucinations on noise/silence
                # come with a high no-speech probability or a very poor
                # average log-probability.
                no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
                if no_speech > 0.6 and avg_logprob < -0.7:
                    logger.debug(
                        "Skipping low-quality segment (no_speech=%.2f logprob=%.2f): %r",
                        no_speech, avg_logprob, text[:60],
                    )
                    continue
                if avg_logprob < -1.2:
                    logger.debug(
                        "Skipping very low-confidence segment (logprob=%.2f): %r",
                        avg_logprob, text[:60],
                    )
                    continue
                out.append(
                    {
                        "text": text,
                        "start_ms": round(seg.start * 1000.0, 1),
                        "end_ms": round(seg.end * 1000.0, 1),
                        "language": lang,
                        "language_probability": lang_prob,
                        "confidence": None,
                    }
                )
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("Whisper transcription failed: %s", exc)
            return []

    def transcribe(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
    ) -> Optional[dict]:
        """Transcribe audio samples (16 kHz mono float32) into a joined result.

        Returns dict with text, start_ms, end_ms, confidence, language,
        or None if transcription failed.
        """
        if self._model is None:
            return None
        if samples.size == 0:
            return None

        result_segments = self.transcribe_segments(samples, language=language)
        if not result_segments:
            return {
                "text": "",
                "start_ms": 0.0,
                "end_ms": round(len(samples) / SAMPLE_RATE * 1000.0, 1),
                "confidence": 0.0,
                "language": None,
                "language_probability": None,
            }

        return {
            "text": " ".join(s["text"] for s in result_segments).strip(),
            "start_ms": result_segments[0]["start_ms"],
            "end_ms": result_segments[-1]["end_ms"],
            "confidence": None,
            "language": result_segments[0].get("language"),
            "language_probability": result_segments[0].get("language_probability"),
        }

    def reset(self) -> None:
        """Reset for a new audio stream (no state in faster-whisper)."""
        pass
