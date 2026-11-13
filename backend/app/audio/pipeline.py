"""Audio processing pipeline: HTTP chunk ingestion -> VAD -> ASR -> Diarization -> Association -> SSE.

Runs a single background asyncio task per session that accumulates audio,
finds speech regions with VAD, transcribes them with Whisper, assigns a
temporary speaker label, then persists and publishes the result over SSE.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MIN_SEGMENT_MS,
    DIARIZATION_THRESHOLD,
    SUMMARY_MIN_SEGMENTS,
    SUMMARY_MAX_SEGMENTS,
    SUMMARY_MIN_SPAN_MS,
    VOICE_ASSOC_CONFIRM_FRAMES,
    VOICE_ASSOC_MAX_OFFSET_MS,
    VOICE_SIMILARITY_THRESHOLD,
)
from .vad import SileroVAD
from .asr import WhisperASR
from .diarize import SpeakerDiarizer, embed
from .associate import VoiceFaceAssociator
from .events import ConversationEventBus
from .summarize import (
    ConversationSummarizer,
    LocalModelSummarizer,
    extract_person_intro,
    join_utterances,
)
from ..db import PeopleDB, EMBEDDING_DIM

logger = logging.getLogger("cantforget.audio.pipeline")

# Processing cadence and buffer limits.
POLL_SECONDS = 0.15
VAD_MIN_INTERVAL_SECONDS = 0.25
MIN_BUFFER_SECONDS = 0.3
TAIL_GUARD_SECONDS = 0.15
# Force-emit very long continuous speech so it does not grow without bound.
MAX_SEGMENT_SECONDS = 6.0
# If VAD finds nothing but the signal is loud, use energy segmentation.
FALLBACK_AFTER_SECONDS = 2.0
# Drop audio older than this when no speech is detected (silence skipping).
SILENCE_ADVANCE_SECONDS = 1.2
SILENCE_RMS = 0.006
FALLBACK_PEAK = 0.05
# Hard cap on retained audio (safety valve for pathological input).
MAX_BUFFER_SECONDS = 30.0
# Common Whisper hallucinations on silence/noise.
_ARTIFACTS = {
    "",
    "[blank_audio]",
    "(silence)",
    "[silence]",
    "(music)",
    "[music]",
    "thank you.",
    "thanks for watching!",
    "thanks for watching.",
}


class AudioPipeline:
    """Processes streaming audio through VAD, ASR, diarization, and association."""

    def __init__(
        self,
        db: PeopleDB,
        event_bus: ConversationEventBus,
        session_id: str,
    ) -> None:
        self.db = db
        self.event_bus = event_bus
        self.session_id = session_id

        # Components (lazily initialized on first audio).
        self._vad: Optional[SileroVAD] = None
        self._asr: Optional[WhisperASR] = None
        self._diarizer: Optional[SpeakerDiarizer] = None
        self._associator: Optional[VoiceFaceAssociator] = None

        # Audio buffer. ``_buf[0]`` corresponds to global sample ``_buf_start``.
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0
        self._buf_lock = asyncio.Lock()
        # Global sample index up to which text has already been emitted.
        self._emitted_until = 0
        self._last_vad_run = 0.0
        self._last_audio = 0.0

        # Pipeline run state.
        self._running = False
        self._process_task: Optional[asyncio.Task] = None

        # Latest face tracks (updated by /api/faces/detect).
        self._face_tracks: list[dict] = []
        self._face_tracks_lock = asyncio.Lock()

        # In-session dedupe for self-introduced persons.
        self._intro_done: set[str] = set()  # speaker ids already assigned a person
        self._person_names: set[str] = set()  # person names already created this session
        self._intro_person: dict[str, str] = {}  # speaker id -> person uid (for late roles)
        # Optional callback (set by the server) fired after a person is
        # auto-created, so the face matcher reloads its gallery and the new
        # face is recognised on the next camera frames.
        self.on_person_created: Optional[object] = None

        # Live per-speaker summaries.
        self._summarizer: Optional[LocalModelSummarizer] = None
        self._summary_pending: list[dict] = []
        self._summary_task: Optional[asyncio.Task] = None
        self._summary_lock = asyncio.Lock()
        # Key points already emitted this session, per person/speaker, so
        # incremental summaries never repeat themselves.
        self._summary_seen: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ setup ---

    async def _ensure_components(self) -> None:
        if self._vad is None:
            self._vad = await asyncio.to_thread(
                SileroVAD,
                threshold=VAD_THRESHOLD,
                min_speech_ms=VAD_MIN_SPEECH_MS,
                min_silence_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            )
            if not self._vad.is_loaded():
                logger.error("VAD not loaded; speech detection unavailable")
        if self._asr is None:
            self._asr = await asyncio.to_thread(
                WhisperASR,
                model_size=WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            if not self._asr.is_loaded():
                logger.error("ASR not loaded; transcription unavailable")
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(
                max_speakers=DIARIZATION_MAX_SPEAKERS,
                similarity_threshold=DIARIZATION_THRESHOLD,
                min_segment_ms=DIARIZATION_MIN_SEGMENT_MS,
            )
        if self._summarizer is None:
            self._summarizer = LocalModelSummarizer()
        if self._associator is None:
            self._associator = VoiceFaceAssociator(
                db=self.db,
                max_offset_ms=VOICE_ASSOC_MAX_OFFSET_MS,
                confirm_frames=VOICE_ASSOC_CONFIRM_FRAMES,
                similarity_threshold=VOICE_SIMILARITY_THRESHOLD,
            )

    # ----------------------------------------------------------------- audio in ---

    async def feed_audio(self, samples: np.ndarray) -> None:
        """Append an audio chunk (16 kHz mono float32) to the buffer."""
        if samples is None:
            return
        chunk = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        async with self._buf_lock:
            if self._buf.size == 0:
                self._buf = chunk.copy()
            else:
                self._buf = np.concatenate((self._buf, chunk))
        self._last_audio = time.monotonic()

    def buffered_seconds(self) -> float:
        return self._buf.size / float(SAMPLE_RATE)

    # ----------------------------------------------------------------- face sync ---

    async def update_face_tracks(self, tracks: list[dict]) -> None:
        """Receive latest face tracks from the detection endpoint."""
        async with self._face_tracks_lock:
            self._face_tracks = tracks

    async def _get_face_tracks(self) -> list[dict]:
        async with self._face_tracks_lock:
            return list(self._face_tracks)

    # ------------------------------------------------------------------- run ---

    async def start(self) -> None:
        """Start the processing loop. Safe to call on every audio POST."""
        if self._process_task is not None and not self._process_task.done():
            return
        try:
            self.db.ensure_session(self.session_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to create session row for %s", self.session_id)
        await self._ensure_components()
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            "Audio pipeline started for session %s | vad=%s asr=%s diarizer=%s associator=%s",
            self.session_id,
            self._vad is not None and self._vad.is_loaded(),
            self._asr is not None and self._asr.is_loaded(),
            self._diarizer is not None,
            self._associator is not None,
        )

    async def stop(self) -> None:
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        # Transcribe any speech still buffered so the last words are not lost.
        try:
            if self._buf.size:
                await self._flush_tail(self._buf, self._buf_start)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to flush audio tail on stop")
        # Summarize whatever was captured before the session ended.
        try:
            for _ in range(4):
                await self._flush_summary(force=True)
                task = self._summary_task
                if task is None or task.done():
                    break
                if task is not asyncio.current_task():
                    await task
        except Exception:  # noqa: BLE001
            logger.exception("Failed to flush conversation summaries on stop")
        logger.info("Audio pipeline stopped for session %s", self.session_id)

    # ------------------------------------------------------------- processing ---

    async def _process_loop(self) -> None:
        """Single loop: accumulate audio, emit finalized speech segments."""
        logger.info("Pipeline loop running for session %s", self.session_id)
        while self._running:
            await asyncio.sleep(POLL_SECONDS)
            try:
                await self._process_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Pipeline iteration failed (session=%s)", self.session_id)

    async def _process_once(self) -> None:
        if self._buf.size < int(SAMPLE_RATE * MIN_BUFFER_SECONDS):
            return
        now = time.monotonic()
        buf = self._buf
        buf_start = self._buf_start

        # The sender paused (e.g. mic stopped): flush whatever remains.
        if now - self._last_audio >= 0.75:
            await self._flush_tail(buf, buf_start)
            return

        if now - self._last_vad_run < VAD_MIN_INTERVAL_SECONDS:
            return
        self._last_vad_run = now

        duration = buf.size / float(SAMPLE_RATE)
        tail_guard = int(SAMPLE_RATE * TAIL_GUARD_SECONDS)

        # 1) Very long continuous speech: force a cut so it gets transcribed.
        if duration >= MAX_SEGMENT_SECONDS:
            local_start = max(0, self._emitted_until - buf_start)
            local_end = buf.size - tail_guard
            if local_end - local_start >= int(SAMPLE_RATE * 0.5):
                await self._emit(buf[local_start:local_end], buf_start + local_start,
                                 buf_start + local_end)
            self._trim(force_advance=True)
            return

        # 2) VAD over the pending buffer.
        segments: list[dict] = []
        if self._vad is not None and self._vad.is_loaded():
            segments = await asyncio.to_thread(self._vad.segment, buf)

        emitted = 0
        for seg in segments:
            local_start = int(round(seg["start_ms"] * SAMPLE_RATE / 1000.0))
            local_end = int(round(seg["end_ms"] * SAMPLE_RATE / 1000.0))
            # Wait for a bit of trailing audio before treating a segment as final,
            # otherwise the last words get cut off.
            if local_end > buf.size - tail_guard:
                continue
            # Clip to the unprocessed region.
            abs_start = buf_start + max(local_start, 0)
            abs_end = buf_start + min(local_end, buf.size)
            if abs_end <= self._emitted_until + int(SAMPLE_RATE * 0.05):
                continue
            clip_start = max(local_start, self._emitted_until - buf_start)
            if local_end - clip_start < int(SAMPLE_RATE * 0.15):
                continue
            await self._emit(buf[clip_start:local_end].copy(),
                             buf_start + clip_start, abs_end)
            emitted += 1

        # 3) No speech detected: skip silence, or fall back to energy VAD.
        if emitted == 0 and not segments:
            peak = float(np.max(np.abs(buf))) if buf.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(buf, dtype=np.float64)))) if buf.size else 0.0
            if duration >= SILENCE_ADVANCE_SECONDS and (rms < SILENCE_RMS or peak < FALLBACK_PEAK):
                logger.debug("Pipeline: skipping %.1fs of silence (rms=%.4f)", duration, rms)
                self._advance_to(buf_start + buf.size - tail_guard)
                self._trim()
                return
            if duration >= FALLBACK_AFTER_SECONDS and peak >= FALLBACK_PEAK:
                logger.info("Pipeline: VAD missed speech, using energy fallback (%.1fs)", duration)
                for s0, s1 in self._energy_segments(buf):
                    abs_start = buf_start + s0
                    abs_end = buf_start + s1
                    if abs_end <= self._emitted_until + int(SAMPLE_RATE * 0.05):
                        continue
                    clip_start = max(s0, self._emitted_until - buf_start)
                    if s1 - clip_start < int(SAMPLE_RATE * 0.2):
                        continue
                    await self._emit(buf[clip_start:s1].copy(),
                                     buf_start + clip_start, abs_end)
                    emitted += 1

        # 4) Drop audio that is already transcribed.
        self._trim()

        if emitted:
            logger.info(
                "Pipeline: emitted %d segment(s) (session=%s, buffered=%.1fs)",
                emitted, self.session_id, self.buffered_seconds(),
            )

    async def _flush_tail(self, buf: np.ndarray, buf_start: int) -> None:
        """Transcribe any remaining speech once audio stops arriving."""
        local_start = max(0, self._emitted_until - buf_start)
        if buf.size - local_start < int(SAMPLE_RATE * 0.25):
            self._trim()
            return
        tail = buf[local_start:]
        peak = float(np.max(np.abs(tail))) if tail.size else 0.0
        if peak < 0.02:
            self._advance_to(buf_start + buf.size)
            self._trim()
            return

        regions: list[tuple[int, int]] = []
        if self._vad is not None and self._vad.is_loaded():
            segs = await asyncio.to_thread(self._vad.segment, tail)
            for seg in segs:
                s0 = int(round(seg["start_ms"] * SAMPLE_RATE / 1000.0))
                s1 = int(round(seg["end_ms"] * SAMPLE_RATE / 1000.0))
                s0 = max(0, min(s0, tail.size))
                s1 = max(0, min(s1, tail.size))
                if s1 - s0 >= int(SAMPLE_RATE * 0.2):
                    regions.append((s0, s1))
        if not regions:
            regions = [(0, tail.size)]

        base = buf_start + local_start
        for s0, s1 in regions:
            abs_start = base + s0
            abs_end = base + s1
            if abs_end <= self._emitted_until + int(SAMPLE_RATE * 0.05):
                continue
            clip_start = max(s0, self._emitted_until - base)
            if s1 - clip_start < int(SAMPLE_RATE * 0.15):
                continue
            await self._emit(tail[clip_start:s1].copy(), base + clip_start, abs_end)
        self._advance_to(buf_start + buf.size)
        self._trim()

    def _energy_segments(self, buf: np.ndarray) -> list[tuple[int, int]]:
        """Simple RMS-based speech regions used only when VAD misses."""
        frame = int(SAMPLE_RATE * 0.03)
        if buf.size < frame:
            return []
        n = buf.size // frame
        frames = buf[: n * frame].reshape(n, frame).astype(np.float32)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        if rms.size == 0:
            return []
        floor = float(np.percentile(rms, 20))
        thresh = max(floor * 3.0, 0.008)
        active = rms >= thresh
        regions: list[tuple[int, int]] = []
        start = None
        gap = 0
        max_gap = int(0.30 / 0.03)  # 300 ms
        for i, is_active in enumerate(active):
            if is_active:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap > max_gap:
                    end = i - gap + 1
                    if end - start >= int(0.20 / 0.03):
                        regions.append((start * frame, end * frame))
                    start = None
                    gap = 0
        if start is not None:
            end = n - gap
            if end - start >= int(0.20 / 0.03):
                regions.append((start * frame, end * frame))
        return regions

    def _advance_to(self, sample_index: int) -> None:
        if sample_index > self._emitted_until:
            self._emitted_until = sample_index

    def _trim(self, force_advance: bool = False) -> None:
        """Drop buffered audio that has already been processed."""
        keep_from = max(self._buf_start, self._emitted_until - int(SAMPLE_RATE * 0.1))
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:].copy()
            self._buf_start = keep_from
        # Safety valve: never keep more than MAX_BUFFER_SECONDS.
        max_samples = int(SAMPLE_RATE * MAX_BUFFER_SECONDS)
        if self._buf.size > max_samples:
            excess = self._buf.size - max_samples
            self._buf = self._buf[excess:].copy()
            self._buf_start += excess
            self._emitted_until = max(self._emitted_until, self._buf_start)

    # -------------------------------------------------------------- transcribe ---

    async def _emit(self, samples: np.ndarray, abs_start: int, abs_end: int) -> None:
        """Transcribe one speech region and publish the result."""
        if samples.size < int(SAMPLE_RATE * 0.15):
            return
        start_ms = abs_start * 1000.0 / SAMPLE_RATE
        end_ms = abs_end * 1000.0 / SAMPLE_RATE
        await self._process_segment(samples, start_ms, end_ms)
        self._advance_to(abs_end)

    async def _process_segment(
        self, seg_samples: np.ndarray, start_ms: float, end_ms: float, depth: int = 0
    ) -> None:
        """Transcribe, diarize, associate, persist, and publish one speech segment.

        Whisper output is split per sentence so each sentence gets its own
        speaker label, instead of labelling a whole multi-sentence region with
        a single voice. If a long region was transcribed as one sentence but
        actually contains a speaker change, it is split at the change point and
        each half is processed separately.
        """
        if seg_samples.size < int(SAMPLE_RATE * 0.15):
            return

        # A long region can hide two speakers (a quick exchange with no real
        # pause). Detect that first and split the audio before ASR, so mixed
        # speech is never transcribed as one voice.
        if depth < 2:
            duration_ms = end_ms - start_ms
            if duration_ms >= 3500:
                split = await asyncio.to_thread(
                    self._detect_speaker_change, seg_samples
                )
                if split is not None:
                    split_ms = split * 1000.0 / SAMPLE_RATE
                    if 1000 <= split_ms <= duration_ms - 1000:
                        logger.info(
                            "Pipeline: splitting %.1fs region at %.2fs (speaker change)",
                            duration_ms / 1000.0, split_ms / 1000.0,
                        )
                        await self._process_segment(
                            seg_samples[:split].copy(),
                            start_ms,
                            start_ms + split_ms,
                            depth=depth + 1,
                        )
                        await self._process_segment(
                            seg_samples[split:].copy(),
                            start_ms + split_ms,
                            end_ms,
                            depth=depth + 1,
                        )
                        return

        # --- ASR (CPU-bound: run off the event loop) ---
        sentences: list[dict] = []
        language = None
        confidence = None
        if self._asr is not None and self._asr.is_loaded():
            language_setting = WHISPER_LANGUAGE
            sentences = await asyncio.to_thread(
                self._asr.transcribe_segments, seg_samples, language_setting
            )
            if sentences:
                language = sentences[0].get("language")
                confidence = sentences[0].get("confidence")

        if not sentences:
            await self._publish_event(
                speaker_id="SPEAKER_UNKNOWN",
                identity_uid=None,
                identity_name=None,
                text="",
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=None,
                language=language,
                associated=False,
                asr_empty=True,
            )
            return

        for sentence in sentences:
            text = (sentence.get("text") or "").strip()
            if not text or text.lower() in _ARTIFACTS:
                continue
            local_start_ms = float(sentence.get("start_ms") or 0.0)
            local_end_ms = float(sentence.get("end_ms") or 0.0)
            abs_start_ms = start_ms + local_start_ms
            abs_end_ms = start_ms + local_end_ms
            if abs_end_ms <= abs_start_ms:
                abs_end_ms = abs_start_ms + 500.0

            # Slice this sentence's audio for speaker embedding / association.
            s0 = max(0, int(round(local_start_ms * SAMPLE_RATE / 1000.0)))
            s1 = min(seg_samples.size, int(round(local_end_ms * SAMPLE_RATE / 1000.0)))
            if s1 - s0 < int(SAMPLE_RATE * 0.15):
                s0, s1 = 0, seg_samples.size
            sentence_audio = seg_samples[s0:s1]

            await self._handle_sentence(
                sentence_audio,
                text,
                abs_start_ms,
                abs_end_ms,
                language=sentence.get("language") or language,
                confidence=sentence.get("confidence", confidence),
            )

    def _detect_speaker_change(self, samples: np.ndarray) -> Optional[int]:
        """Return a sample index where the speaker likely changes, or None.

        Slides 1.5 s windows with a 0.75 s hop and embeds each window. The first
        and last windows seed the two possible voices; every window is assigned
        to the nearer seed and a single transition becomes the split point.
        Conservative by design: a false split would cut a sentence in half, so
        both sides must be internally consistent and clearly different.
        """
        sr = SAMPLE_RATE
        n = samples.size
        if n < int(sr * 3.5):
            return None
        win = int(sr * 1.5)
        hop = int(sr * 0.75)

        embs: list[np.ndarray] = []
        centers: list[int] = []
        pos = 0
        while pos + win <= n:
            vec = embed(samples[pos:pos + win])
            if vec is None:
                return None
            embs.append(vec)
            centers.append(pos + win // 2)
            pos += hop

        if len(embs) < 5:
            return None

        def unit(v: np.ndarray) -> np.ndarray:
            return v / max(float(np.linalg.norm(v)), 1e-8)

        left_seed = unit(np.mean(embs[:2], axis=0))
        right_seed = unit(np.mean(embs[-2:], axis=0))
        # The two ends of the region must actually sound different.
        if float(np.dot(left_seed, right_seed)) > 0.35:
            return None

        assigned = [
            0 if float(np.dot(e, left_seed)) >= float(np.dot(e, right_seed)) else 1
            for e in embs
        ]
        transitions = [i for i in range(1, len(assigned)) if assigned[i] != assigned[i - 1]]
        if len(transitions) != 1:
            return None
        k = transitions[0]
        if k < 2 or len(embs) - k < 2:
            return None

        left_mean = unit(np.mean(embs[:k], axis=0))
        right_mean = unit(np.mean(embs[k:], axis=0))
        if float(np.dot(left_mean, right_mean)) > 0.30:
            return None

        split = (centers[k - 1] + centers[k]) // 2
        if split < int(sr * 0.8) or n - split < int(sr * 0.8):
            return None
        return int(split)

    async def _handle_sentence(
        self,
        samples: np.ndarray,
        text: str,
        start_ms: float,
        end_ms: float,
        language: Optional[str],
        confidence: Optional[float],
    ) -> None:
        """Diarize, associate, persist, and publish one sentence."""
        # --- Diarization (temporary labels only) ---
        speaker_id = "SPEAKER_UNKNOWN"
        if self._diarizer is not None:
            speaker_id = await asyncio.to_thread(
                self._diarizer.assign, samples, max(0.0, end_ms - start_ms)
            )

        # --- Voice-to-face association (only for enrolled + recognized faces) ---
        identity_uid: Optional[str] = None
        assoc_confidence = None
        if self._associator is not None:
            face_tracks = await self._get_face_tracks()
            linked_uid = self._associator.associate_segment(
                session_id=self.session_id,
                speaker_id=speaker_id,
                samples=samples,
                start_ms=start_ms,
                end_ms=end_ms,
                active_face_tracks=face_tracks,
            )
            if linked_uid:
                identity_uid = linked_uid
                assoc_confidence = self._associator.similarity_threshold
        if identity_uid is None:
            # Reuse a previously persisted speaker->person link (e.g. from a
            # self-introduction earlier in this session) so later sentences
            # are attributed to the same person.
            try:
                identity_uid = self.db.get_speaker_identity(self.session_id, speaker_id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to look up speaker identity")

        identity_name: Optional[str] = None

        # --- Self-introduction: "hey im X, im your son" ---
        # If the voice is not yet linked to a face-recognised person, listen for a
        # self-introduction and auto-create a provisional person (name + role) from
        # the speech, enrol the current voice segment, and link the speaker to them.
        # If a face is visible during the intro, also enrol the face embedding so the
        # person can be recognised on camera later.
        intro = (
            extract_person_intro(text)
            if speaker_id != "SPEAKER_UNKNOWN" and text
            else None
        )
        if identity_uid is None and intro and speaker_id not in self._intro_done:
            name = intro.get("name")
            relationship = intro.get("relationship", "")
            if name and name not in self._person_names:
                # Pick the best visible face for this moment: the largest
                # track (closest to the camera) that is not confidently
                # someone else. The voice has not been linked to a
                # recognised face, so any visible face is a candidate.
                face_tracks = await self._get_face_tracks()
                best_track = None
                for track in face_tracks:
                    if track.get("embedding") is None and track.get("face_crop") is None:
                        continue
                    t_uid = track.get("person_uid")
                    if t_uid is not None and t_uid != "new":
                        continue
                    area = float(track.get("w", 0)) * float(track.get("h", 0))
                    if best_track is None or area > float(best_track.get("w", 0)) * float(best_track.get("h", 0)):
                        best_track = track

                face_vec = None
                face_crop = None
                if best_track is not None:
                    t_emb = best_track.get("embedding")
                    if t_emb:
                        vec = np.frombuffer(t_emb, dtype=np.float32)
                        if vec.shape[0] == EMBEDDING_DIM:
                            face_vec = vec
                    face_crop = best_track.get("face_crop")

                existing = self.db.get_person_by_name(name)
                if existing:
                    person_uid = existing["uid"]
                    # Backfill thumbnail for an existing speech-intro person.
                    if face_crop and not existing.get("has_thumbnail", True):
                        self.db.set_person_thumbnail(person_uid, face_crop)
                else:
                    person = self.db.create_person_provisional(
                        name=name,
                        tag=relationship,
                        description=f"Introduced by voice: \"{text}\"",
                        thumbnail=face_crop,
                    )
                    person_uid = person["uid"]
                    self._person_names.add(name)
                # Enrol this voice segment as the person's voice profile.
                try:
                    fp = await asyncio.to_thread(embed, samples)
                    if fp is None:
                        raise RuntimeError("no speaker embedding")
                    now_iso = datetime.now(timezone.utc).isoformat()
                    self.db.add_voice_enrollment(
                        person_uid=person_uid,
                        embedding=fp,
                        consent_at=now_iso,
                        model_version="eres2net:512d",
                    )
                    # Enrol the face embedding captured from the camera so
                    # the person is recognised next time they appear.
                    if face_vec is not None:
                        self.db.add_embeddings(person_uid, [face_vec])
                        logger.info(
                            "Pipeline: self-intro %s enrolled face embedding from track %s",
                            speaker_id, best_track.get("track_id"),
                        )
                    else:
                        logger.debug(
                            "Pipeline: no usable face embedding for self-intro %s; voice only",
                            speaker_id,
                        )
                    # Reload the face matcher gallery so the new face is
                    # recognised on subsequent camera frames.
                    if self.on_person_created:
                        try:
                            self.on_person_created()
                        except Exception:  # noqa: BLE001
                            logger.exception("Failed to reload identity gallery")
                    self.db.link_speaker_to_identity(
                        session_id=self.session_id,
                        speaker_id=speaker_id,
                        identity_uid=person_uid,
                        confidence=0.0,
                        method="self_intro",
                    )
                    identity_uid = person_uid
                    identity_name = name
                    self._intro_done.add(speaker_id)
                    self._intro_person[speaker_id] = person_uid
                    logger.info(
                        "Pipeline: self-intro %s -> %s (%s) from %r",
                        speaker_id, person_uid, name, text[:80],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to enrol self-intro voice: %s", exc)
        elif (
            intro
            and intro.get("relationship")
            and speaker_id in self._intro_done
            and self._intro_person.get(speaker_id)
        ):
            # The name was said earlier, but the role ("...im your
            # son") came in a later sentence. Fill in the empty tag.
            person_uid = self._intro_person[speaker_id]
            person = self.db.get_person(person_uid)
            if person and not (person.get("tag") or "").strip():
                try:
                    self.db.update_person(
                        person_uid,
                        person["name"],
                        intro["relationship"],
                        person["description"] or "",
                    )
                    logger.info(
                        "Pipeline: late role %r attached to %s from %r",
                        intro["relationship"], person_uid, text[:80],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to attach late role: %s", exc)

        # --- Persist ---
        seg_id = None
        try:
            seg_id = self.db.add_transcription_segment(
                session_id=self.session_id,
                speaker_id=speaker_id,
                text=text,
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                confidence=confidence,
                language=language,
                identity_uid=identity_uid,
            )
            if identity_uid:
                self.db.link_speaker_to_identity(
                    session_id=self.session_id,
                    speaker_id=speaker_id,
                    identity_uid=identity_uid,
                    confidence=assoc_confidence or 0.0,
                    method="timestamp_sync",
                )
                person = self.db.get_person(identity_uid)
                if person:
                    identity_name = person.get("name")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to persist transcription segment: %s", exc)

        # --- Publish via SSE ---
        await self._publish_event(
            speaker_id=speaker_id,
            identity_uid=identity_uid,
            identity_name=identity_name,
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=confidence,
            language=language,
            associated=identity_uid is not None,
            asr_empty=False,
            segment_id=seg_id,
        )
        logger.info(
            "Pipeline: [%s] %s -> %r (%.1fs-%.1fs)",
            self.session_id, speaker_id, text[:80], start_ms / 1000.0, end_ms / 1000.0,
        )

        # --- Queue for a gentle per-speaker summary ---
        self._summary_pending.append(
            {
                "speaker_id": speaker_id,
                "identity_uid": identity_uid,
                "identity_name": identity_name,
                "text": text,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
            }
        )
        self._maybe_schedule_summary()

    async def _publish_event(
        self,
        speaker_id: str,
        identity_uid: Optional[str],
        identity_name: Optional[str],
        text: str,
        start_ms: float,
        end_ms: float,
        confidence: Optional[float],
        language: Optional[str],
        associated: bool,
        asr_empty: bool,
        segment_id: Optional[str] = None,
    ) -> None:
        """Publish a transcription event via SSE."""
        event = {
            "event_type": "transcription",
            "session_id": self.session_id,
            "segment_id": segment_id,
            "speaker_id": speaker_id,
            "identity_id": identity_uid,
            "identity_name": identity_name,
            "text": text,
            "start_ms": round(start_ms, 1),
            "end_ms": round(end_ms, 1),
            "confidence": confidence,
            "language": language,
            "associated": associated,
            "asr_empty": asr_empty,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.event_bus.publish(self.session_id, event)

    # --------------------------------------------------------------- summaries ---

    def _maybe_schedule_summary(self) -> None:
        """Kick off a summary when enough new speech has accumulated."""
        if self._summary_task is not None and not self._summary_task.done():
            return
        if not self._summary_pending:
            return

        batch = self._summary_pending
        segment_count = len(batch)
        span_ms = batch[-1]["end_ms"] - batch[0]["start_ms"]
        if segment_count >= SUMMARY_MAX_SEGMENTS:
            ready = True
        elif segment_count >= SUMMARY_MIN_SEGMENTS and span_ms >= SUMMARY_MIN_SPAN_MS:
            ready = True
        else:
            ready = False
        if ready:
            self._summary_task = asyncio.create_task(self._run_summary())

    async def _run_summary(self) -> None:
        """Background worker: one summary pass, then reschedule if needed."""
        try:
            await self._flush_summary()
        except Exception:  # noqa: BLE001
            logger.exception("Conversation summary pass failed")
        finally:
            self._summary_task = None
            if self._summary_pending:
                self._maybe_schedule_summary()

    async def _flush_summary(self, force: bool = False) -> None:
        """Summarize pending utterances per speaker and persist the key points.

        The lock serializes passes, so a stop-time flush can safely wait for an
        in-flight background pass instead of awaiting its own task.
        """
        async with self._summary_lock:
            if not self._summary_pending:
                return
            if not force and len(self._summary_pending) < SUMMARY_MIN_SEGMENTS:
                return
            batch, self._summary_pending = self._summary_pending, []
            await self._summarize_batch(batch)

    async def _summarize_batch(self, batch: list[dict]) -> None:
        if not batch or self._summarizer is None:
            return

        # Canonicalize: if a speaker was linked to a person at any point in the
        # batch (or earlier in the session), group all of that speaker's
        # sentences under the person.
        db_links: dict[str, str] = {}
        try:
            for assoc in self.db.get_voice_associations(self.session_id):
                uid = assoc.get("identity_uid")
                spk = assoc.get("speaker_id")
                if uid and spk:
                    db_links[spk] = uid
        except Exception:  # noqa: BLE001
            logger.exception("Could not load voice associations for summaries")

        speaker_to_identity: dict[str, str] = dict(db_links)
        for seg in batch:
            if seg.get("identity_uid"):
                speaker_to_identity[seg["speaker_id"]] = seg["identity_uid"]

        groups: dict[str, list[dict]] = {}
        for seg in batch:
            speaker_id = seg["speaker_id"]
            key = speaker_to_identity.get(speaker_id) or speaker_id
            groups.setdefault(key, []).append(seg)

        summarizer = self._summarizer
        for key, segs in groups.items():
            text = join_utterances([s["text"] for s in segs if s.get("text")])
            if not text:
                continue
            try:
                points = await asyncio.to_thread(summarizer.summarize, text)
            except Exception:  # noqa: BLE001
                logger.exception("Summary generation failed for %s", key)
                continue
            if not points:
                continue

            # Suppress points already emitted for this person this session.
            seen = self._summary_seen.setdefault(key, [])
            fresh = [
                p for p in points
                if p and not ConversationSummarizer._is_near_duplicate(p, seen, 0.55)
            ]
            if not fresh:
                continue
            seen.extend(fresh)
            del seen[:-20]
            points = fresh

            identity_uid = segs[0].get("identity_uid") or speaker_to_identity.get(
                segs[0]["speaker_id"]
            )
            identity_name = segs[0].get("identity_name")
            speaker_id = segs[0]["speaker_id"]
            if identity_uid and not identity_name:
                try:
                    person = self.db.get_person(identity_uid)
                    if person:
                        identity_name = person.get("name")
                except Exception:  # noqa: BLE001
                    pass
            start_ms = min(s["start_ms"] for s in segs)
            end_ms = max(s["end_ms"] for s in segs)
            model_version = summarizer.model_version

            summary_id = None
            try:
                summary_id = self.db.add_conversation_summary(
                    session_id=self.session_id,
                    speaker_id=speaker_id,
                    key_points=points,
                    identity_uid=identity_uid,
                    summary_text="\n".join(points),
                    model_version=model_version,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    segment_count=len(segs),
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist conversation summary")

            await self.event_bus.publish(
                self.session_id,
                {
                    "event_type": "summary",
                    "session_id": self.session_id,
                    "summary_id": summary_id,
                    "speaker_id": speaker_id,
                    "identity_id": identity_uid,
                    "identity_name": identity_name,
                    "key_points": points,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "segment_count": len(segs),
                    "model_version": model_version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(
                "Pipeline: summary [%s] %s -> %d point(s) (%s)",
                self.session_id, speaker_id, len(points), model_version,
            )
