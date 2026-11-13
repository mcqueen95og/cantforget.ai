"""Conversation Intelligence Engine — Phase 4.

Transforms completed conversation transcripts into structured summaries and
candidate personal-memory records, with strict attribution and approval rules.

The engine is intentionally independent of the existing speech-recognition
pipeline: it consumes *persisted transcript segments* (text + speaker + time +
identity), never raw audio.

Design constraints for the target hardware (i3 / 8 GB CPU-only):
  * No large model is required by default. With ``LLM_PROVIDER=none`` (the
    default) a lightweight, dependency-free extractive path is used.
  * Optional LLM backends (Ollama, OpenAI, or a custom OpenAI-compatible URL)
    are enabled through environment variables.
  * All model output is validated against a strict JSON schema before being
    persisted. Malformed output is never written to the database.

Privacy & safety rules (non-negotiable):
  * Only *explicitly stated* information becomes a persistent memory.
  * Never infer medical, mental-health, religious, political, sexual-orientation
    or financial attributes.
  * Never build personality profiles from voice, face or conversational style.
  * Unknown / uncertain speakers are never assigned a name.
  * Every extracted record keeps its original statement, speaker, session and
    model version for audit and participant review.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..db import PeopleDB
from .summarize import ConversationSummarizer, extract_person_intro

logger = logging.getLogger("cantforget.audio.intelligence")

# --------------------------------------------------------------------------- config ---

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")  # none | ollama | openai | custom
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "30.0"))

# Conversation windowing.
CONVERSATION_INACTIVITY_MS = int(os.environ.get("CONVERSATION_INACTIVITY_MS", "30000"))
CONVERSATION_MIN_SEGMENTS = int(os.environ.get("CONVERSATION_MIN_SEGMENTS", "2"))
CONVERSATION_MAX_WINDOW_SECONDS = int(os.environ.get("CONVERSATION_MAX_WINDOW_SECONDS", "600"))

# Information-extraction safety.
SENSITIVE_PATTERNS = [
    re.compile(r"\b(?:cancer|diabetes|depression|anxiety|bipolar|schizophrenia|ptsd|dementia|alzheimer)\b", re.I),
    re.compile(r"\b(?:muslim|christian|jewish|hindu|buddhist|atheist|agnostic|religion)\b", re.I),
    re.compile(r"\b(?:republican|democrat|conservative|liberal|socialist|politics|election|vote)\b", re.I),
    re.compile(r"\b(?:gay|lesbian|bisexual|transgender|sexual|homosexual)\b", re.I),
    re.compile(r"\b(?:salary|wage|debt|bankrupt|credit score|net worth|inheritance)\b", re.I),
]

ALLOWED_CATEGORIES = {
    "fact", "interest", "preference", "project", "commitment", "decision",
}

DEFAULT_MODEL_VERSION = "cantforget-intelligence:v1"


# --------------------------------------------------------------------------- data ---

@dataclass
class TranscriptSegment:
    """A single transcribed utterance."""
    speaker_id: str
    identity_uid: Optional[str]
    identity_name: Optional[str]
    text: str
    start_ms: int
    end_ms: int
    confidence: Optional[float] = None
    verification_status: str = "unknown"  # verified | uncertain | unknown


@dataclass
class ConversationWindow:
    """A chronological window of transcript segments ready for processing."""
    window_id: str
    session_id: str
    window_index: int
    segments: list[TranscriptSegment]
    started_at: str
    ended_at: Optional[str]
    status: str = "open"


@dataclass
class ExtractedRecord:
    """A candidate extracted-information record awaiting approval."""
    speaker_id: str
    participant_uid: Optional[str]
    participant_name: Optional[str]
    category: str
    normalized_info: str
    original_statement: str
    statement_start_ms: Optional[int]
    statement_end_ms: Optional[int]
    confidence: Optional[float]
    model_version: str


@dataclass
class ConversationSummary:
    """A structured summary of a conversation window."""
    speaker_id: str
    identity_uid: Optional[str]
    identity_name: Optional[str]
    summary_text: str
    topics: list[str]
    decisions: list[str]
    key_points: list[str]
    model_version: str


# --------------------------------------------------------------------------- LLM ---

class LLMClient:
    """Abstracts a chat/completion model that returns structured JSON.

    With ``LLM_PROVIDER=none`` the client is a no-op and callers should use
    the extractive fallback. Other providers are best-effort: if the model is
    unavailable or returns invalid JSON, the caller falls back to extractive.
    """

    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY
        self.timeout = LLM_TIMEOUT_SECONDS
        self._available = self.provider != "none"
        if self.provider == "none":
            logger.info("Intelligence LLM disabled; using extractive fallback")
        else:
            logger.info(
                "Intelligence LLM configured: provider=%s model=%s base=%s",
                self.provider, self.model, self.base_url,
            )

    def is_available(self) -> bool:
        return self._available

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Optional[dict]:
        """Return parsed JSON from the model, or None on any failure."""
        if not self._available:
            return None
        try:
            import aiohttp  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("aiohttp not installed; LLM provider %s unavailable", self.provider)
            self._available = False
            return None

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if self.provider == "openai" or self.provider == "custom":
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "ollama":
            url = f"{self.base_url.rstrip('/')}/api/chat"
            headers = {"Content-Type": "application/json"}
        else:
            logger.warning("Unknown LLM provider %s", self.provider)
            return None

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning("LLM returned HTTP %s", resp.status)
                        return None
                    data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM request failed: %s", exc)
            return None

        # Extract text depending on provider shape.
        text: Optional[str] = None
        try:
            if self.provider == "ollama":
                text = data.get("message", {}).get("content")
            else:
                text = data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:  # noqa: BLE001
            return None
        if not text:
            return None

        # Strip markdown fences if present.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:  # noqa: BLE001
            logger.warning("LLM returned non-JSON: %r", text[:200])
            return None


# --------------------------------------------------------------------------- engine ---

class ConversationIntelligenceEngine:
    """Orchestrates windowing, summarization, extraction and persistence."""

    def __init__(self, db: PeopleDB, llm: Optional[LLMClient] = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()
        self._summarizer = ConversationSummarizer()
        # In-memory session state: session_id -> list of pending segments.
        self._pending: dict[str, list[TranscriptSegment]] = {}
        # Last segment end time per session, for windowing.
        self._last_end_ms: dict[str, int] = {}

    # ------------------------------------------------------------------ ingestion ---

    def add_segment(
        self,
        session_id: str,
        speaker_id: str,
        identity_uid: Optional[str],
        identity_name: Optional[str],
        text: str,
        start_ms: int,
        end_ms: int,
        confidence: Optional[float] = None,
        verification_status: str = "unknown",
    ) -> None:
        """Append a transcript segment to a session's pending buffer."""
        seg = TranscriptSegment(
            speaker_id=speaker_id,
            identity_uid=identity_uid,
            identity_name=identity_name,
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=confidence,
            verification_status=verification_status,
        )
        self._pending.setdefault(session_id, []).append(seg)
        self._last_end_ms[session_id] = end_ms

    def get_pending_segments(self, session_id: str) -> list[TranscriptSegment]:
        """Return segments not yet assigned to a finalized window."""
        return list(self._pending.get(session_id, []))

    def should_finalize(self, session_id: str, now_ms: int) -> bool:
        """Decide whether the current window should be finalized."""
        segs = self._pending.get(session_id, [])
        if not segs:
            return False
        if len(segs) < CONVERSATION_MIN_SEGMENTS:
            return False
        # Window too long: force a cut.
        if segs and (now_ms - segs[0].start_ms) >= CONVERSATION_MAX_WINDOW_SECONDS * 1000:
            return True
        # Inactivity gap since the last segment.
        if now_ms - self._last_end_ms.get(session_id, 0) >= CONVERSATION_INACTIVITY_MS:
            return True
        return False

    def build_window(self, session_id: str) -> Optional[ConversationWindow]:
        """Finalize and return the current pending window for a session."""
        segs = self._pending.get(session_id, [])
        if not segs:
            return None
        # Ensure chronological order.
        segs = sorted(segs, key=lambda s: s.start_ms)
        # Determine window index.
        existing = self.db.get_windows(session_id, limit=1)
        window_index = (existing[0]["window_index"] + 1) if existing else 0
        window_id = "cw_" + uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        window = ConversationWindow(
            window_id=window_id,
            session_id=session_id,
            window_index=window_index,
            segments=segs,
            started_at=started_at,
            ended_at=None,
        )
        # Persist the window as 'open'.
        self.db.create_conversation_window(session_id, window_index, started_at)
        # Clear the pending buffer.
        self._pending[session_id] = []
        self._last_end_ms[session_id] = 0
        return window

    # ----------------------------------------------------------- extraction rules ---

    @staticmethod
    def _is_sensitive(text: str) -> bool:
        """Return True if the statement touches a prohibited attribute category."""
        return any(p.search(text) for p in SENSITIVE_PATTERNS)

    @staticmethod
    def _validate_category(category: str) -> bool:
        return category in ALLOWED_CATEGORIES

    def _filter_extracted(
        self, records: list[dict], session_id: str
    ) -> list[ExtractedRecord]:
        """Validate and sanitize model-extracted records.

        Applies the strict information-extraction rules:
          * Only allowed categories.
          * No sensitive attributes.
          * Original statement preserved.
          * Speaker attribution preserved (verified / uncertain / unknown).
        """
        out: list[ExtractedRecord] = []
        for r in records:
            category = str(r.get("category", "")).strip().lower()
            if not self._validate_category(category):
                logger.debug("Dropping extracted record with disallowed category %r", category)
                continue
            normalized = str(r.get("normalized_info", "")).strip()
            statement = str(r.get("original_statement", "")).strip()
            if not normalized or not statement:
                continue
            if self._is_sensitive(normalized) or self._is_sensitive(statement):
                logger.debug("Dropping sensitive extracted record: %r", statement[:80])
                continue
            speaker_id = str(r.get("speaker_id", "SPEAKER_UNKNOWN"))
            participant_uid = r.get("participant_uid")
            participant_name = r.get("participant_name")
            # Never fabricate a participant link.
            if participant_uid is not None and not str(participant_uid).startswith("p_"):
                participant_uid = None
                participant_name = None
            out.append(
                ExtractedRecord(
                    speaker_id=speaker_id,
                    participant_uid=participant_uid,
                    participant_name=participant_name,
                    category=category,
                    normalized_info=normalized,
                    original_statement=statement,
                    statement_start_ms=r.get("statement_start_ms"),
                    statement_end_ms=r.get("statement_end_ms"),
                    confidence=r.get("confidence"),
                    model_version=DEFAULT_MODEL_VERSION,
                )
            )
        return out

    def _deduplicate(
        self,
        records: list[ExtractedRecord],
        session_id: str,
    ) -> list[ExtractedRecord]:
        """Conservative deduplication against existing approved/pending records.

        Exact-match (participant + category + normalized_info) is suppressed.
        Conflicts (same participant/category, different normalized_info) are kept
        for review rather than silently overwritten.
        """
        existing = self.db.get_extracted_information(session_id, limit=500)
        seen: set[tuple[str, str, str]] = set()
        for rec in existing:
            key = (
                str(rec.get("participant_uid") or ""),
                str(rec.get("category")),
                str(rec.get("normalized_info")).lower(),
            )
            seen.add(key)

        out: list[ExtractedRecord] = []
        for rec in records:
            key = (
                rec.participant_uid or "",
                rec.category,
                rec.normalized_info.lower(),
            )
            if key in seen:
                logger.debug("Suppressing duplicate extracted record: %r", rec.normalized_info)
                continue
            seen.add(key)
            out.append(rec)
        return out

    # ------------------------------------------------------------- LLM prompts ---

    @staticmethod
    def _summarization_prompt(window: ConversationWindow) -> tuple[str, str]:
        lines = []
        for s in window.segments:
            tag = s.identity_name or s.speaker_id
            lines.append(f"[{s.start_ms}ms-{s.end_ms}ms] {tag}: {s.text}")
        transcript = "\n".join(lines)

        system = (
            "You are a careful conversation-intelligence assistant. "
            "You receive a transcript with speaker labels and timestamps. "
            "Return ONLY a JSON object with keys: summary_text, topics, decisions, key_points. "
            "summary_text: a concise 2-4 sentence summary of what was discussed. "
            "topics: a list of major discussion topics (short phrases). "
            "decisions: a list of explicit decisions made (empty list if none). "
            "key_points: a list of short gentle key points per speaker. "
            "Do not infer sensitive attributes (health, religion, politics, sexuality, finances). "
            "Do not invent names. Use only speaker labels from the transcript."
        )
        user = (
            "Transcript:\n" + transcript + "\n\n"
            "Return the JSON object now."
        )
        return system, user

    @staticmethod
    def _extraction_prompt(window: ConversationWindow) -> tuple[str, str]:
        lines = []
        for s in window.segments:
            tag = s.identity_name or s.speaker_id
            lines.append(f"[{s.start_ms}ms-{s.end_ms}ms] {tag}: {s.text}")
        transcript = "\n".join(lines)

        system = (
            "You extract explicitly stated personal information from a conversation. "
            "Return ONLY a JSON array of objects. Each object has keys: "
            "speaker_id, category, normalized_info, original_statement, statement_start_ms, statement_end_ms, confidence. "
            "category must be one of: fact, interest, preference, project, commitment, decision. "
            "normalized_info: the fact/interest/preference in normalized form. "
            "original_statement: the exact words spoken. "
            "Only include information the speaker explicitly stated. "
            "NEVER include: health conditions, mental health, religion, politics, sexual orientation, financial status. "
            "NEVER invent a name or identity. Use speaker_id exactly as given. "
            "If nothing explicit is stated, return an empty array."
        )
        user = (
            "Transcript:\n" + transcript + "\n\n"
            "Return the JSON array now."
        )
        return system, user

    # --------------------------------------------------------------- processing ---

    async def process_window(self, window: ConversationWindow) -> dict:
        """Run summarization + extraction over a finalized window.

        Returns a dict with summary and extracted records (as dicts).
        All database writes happen inside transactions.
        """
        session_id = window.session_id

        # 1. Build attribution map from existing voice associations.
        assocs = self.db.get_voice_associations(session_id)
        assoc_map: dict[str, str] = {}
        for a in assocs:
            assoc_map[a["speaker_id"]] = a["identity_uid"]

        # 2. Summarize (LLM with extractive fallback).
        summary = await self._summarize_window(window, assoc_map)

        # 3. Extract information (LLM with extractive fallback).
        records = await self._extract_information(window, assoc_map)

        # 4. Validate, filter, deduplicate.
        records = self._filter_extracted([r.__dict__ for r in records], session_id)
        records = self._deduplicate(records, session_id)

        # 5. Persist summary.
        summary_id = self.db.add_conversation_summary(
            session_id=session_id,
            speaker_id=summary.speaker_id,
            key_points=summary.key_points,
            identity_uid=summary.identity_uid,
            summary_text=summary.summary_text,
            topics=summary.topics,
            decisions=summary.decisions,
            model_version=summary.model_version,
            window_id=window.window_id,
        )

        # 6. Persist extracted records (pending approval).
        saved_ids: list[str] = []
        for rec in records:
            rid = self.db.add_extracted_information(
                session_id=session_id,
                speaker_id=rec.speaker_id,
                category=rec.category,
                normalized_info=rec.normalized_info,
                original_statement=rec.original_statement,
                model_version=rec.model_version,
                participant_uid=rec.participant_uid,
                window_id=window.window_id,
                statement_start_ms=rec.statement_start_ms,
                statement_end_ms=rec.statement_end_ms,
                confidence=rec.confidence,
            )
            saved_ids.append(rid)

        # 7. Mark window completed.
        with self.db._lock:  # noqa: SLF001 - internal coordinated write
            self.db._conn.execute(
                "UPDATE conversation_windows SET status='completed', ended_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), window.window_id),
            )
            self.db._conn.commit()

        return {
            "window_id": window.window_id,
            "session_id": session_id,
            "summary_id": summary_id,
            "summary": {
                "speaker_id": summary.speaker_id,
                "identity_uid": summary.identity_uid,
                "identity_name": summary.identity_name,
                "summary_text": summary.summary_text,
                "topics": summary.topics,
                "decisions": summary.decisions,
                "key_points": summary.key_points,
                "model_version": summary.model_version,
            },
            "extracted": [
                {
                    "id": rid,
                    "speaker_id": r.speaker_id,
                    "participant_uid": r.participant_uid,
                    "participant_name": r.participant_name,
                    "category": r.category,
                    "normalized_info": r.normalized_info,
                    "original_statement": r.original_statement,
                    "statement_start_ms": r.statement_start_ms,
                    "statement_end_ms": r.statement_end_ms,
                    "confidence": r.confidence,
                    "verification_status": "pending",
                    "model_version": r.model_version,
                }
                for rid, r in zip(saved_ids, records)
            ],
        }

    async def _summarize_window(
        self, window: ConversationWindow, assoc_map: dict[str, str]
    ) -> ConversationSummary:
        system, user = self._summarization_prompt(window)

        # LLM attempt.
        if self.llm.is_available():
            data = await self.llm.chat_json(system, user)
            if data:
                return ConversationSummary(
                    speaker_id=window.segments[0].speaker_id if window.segments else "SPEAKER_UNKNOWN",
                    identity_uid=assoc_map.get(window.segments[0].speaker_id) if window.segments else None,
                    identity_name=None,
                    summary_text=str(data.get("summary_text", ""))[:2000],
                    topics=[str(t) for t in (data.get("topics") or [])[:10]],
                    decisions=[str(d) for d in (data.get("decisions") or [])[:10]],
                    key_points=[str(k) for k in (data.get("key_points") or [])[:20]],
                    model_version=f"llm:{LLM_MODEL}",
                )

        # Extractive fallback.
        by_speaker: dict[str, list[str]] = {}
        for s in window.segments:
            by_speaker.setdefault(s.speaker_id, []).append(s.text)
        points_by_speaker = self._summarizer.summarize_by_person(by_speaker)
        all_points: list[str] = []
        for pts in points_by_speaker.values():
            all_points.extend(pts)

        # Detect self-introductions for provisional person creation.
        primary_speaker = window.segments[0].speaker_id if window.segments else "SPEAKER_UNKNOWN"
        identity_uid = assoc_map.get(primary_speaker)
        identity_name = None
        if identity_uid:
            person = self.db.get_person(identity_uid)
            if person:
                identity_name = person.get("name")

        summary_text = " ".join(all_points[:6]) if all_points else "No significant content detected."
        topics: list[str] = []
        for s in window.segments:
            intro = extract_person_intro(s.text)
            if intro and intro.get("name"):
                topics.append(f"self_introduction: {intro['name']}")
        topics = list(dict.fromkeys(topics))[:5]

        return ConversationSummary(
            speaker_id=primary_speaker,
            identity_uid=identity_uid,
            identity_name=identity_name,
            summary_text=summary_text,
            topics=topics,
            decisions=[],
            key_points=all_points[:10],
            model_version=DEFAULT_MODEL_VERSION,
        )

    async def _extract_information(
        self, window: ConversationWindow, assoc_map: dict[str, str]
    ) -> list[ExtractedRecord]:
        system, user = self._extraction_prompt(window)

        raw_records: list[dict] = []
        if self.llm.is_available():
            data = await self.llm.chat_json(system, user)
            if isinstance(data, list):
                raw_records = [r for r in data if isinstance(r, dict)]
            elif isinstance(data, dict) and isinstance(data.get("records"), list):
                raw_records = data["records"]

        records: list[ExtractedRecord] = []
        for r in raw_records:
            speaker_id = str(r.get("speaker_id", "SPEAKER_UNKNOWN"))
            # Resolve participant identity strictly through existing associations.
            participant_uid = assoc_map.get(speaker_id)
            participant_name = None
            if participant_uid:
                person = self.db.get_person(participant_uid)
                if person:
                    participant_name = person.get("name")
            records.append(
                ExtractedRecord(
                    speaker_id=speaker_id,
                    participant_uid=participant_uid,
                    participant_name=participant_name,
                    category=str(r.get("category", "fact")).strip().lower(),
                    normalized_info=str(r.get("normalized_info", "")).strip(),
                    original_statement=str(r.get("original_statement", "")).strip(),
                    statement_start_ms=r.get("statement_start_ms"),
                    statement_end_ms=r.get("statement_end_ms"),
                    confidence=r.get("confidence"),
                    model_version=f"llm:{LLM_MODEL}" if self.llm.is_available() else DEFAULT_MODEL_VERSION,
                )
            )

        # Extractive fallback: capture self-introductions as facts.
        if not records:
            for s in window.segments:
                intro = extract_person_intro(s.text)
                if intro:
                    participant_uid = assoc_map.get(s.speaker_id)
                    participant_name = None
                    if participant_uid:
                        person = self.db.get_person(participant_uid)
                        if person:
                            participant_name = person.get("name")
                    parts = []
                    if intro.get("name"):
                        parts.append(f"name: {intro['name']}")
                    if intro.get("relationship"):
                        parts.append(f"relationship: {intro['relationship']}")
                    records.append(
                        ExtractedRecord(
                            speaker_id=s.speaker_id,
                            participant_uid=participant_uid,
                            participant_name=participant_name,
                            category="fact",
                            normalized_info="; ".join(parts),
                            original_statement=s.text,
                            statement_start_ms=s.start_ms,
                            statement_end_ms=s.end_ms,
                            confidence=None,
                            model_version=DEFAULT_MODEL_VERSION,
                        )
                    )
        return records
