"""Conversation summarization: small, gentle key points per speaker/person.

Distills a conversation's transcript segments into a handful of short key
points, grouped by who said what (diarized speaker label, linked to a
recognised face identity where available).

Designed for CPU-only, low-RAM hardware (the i3/8GB target). The default
``ConversationSummarizer`` is extractive and dependency-free beyond the stdlib:
it ranks sentences by term frequency and keeps the most informative ones, so
nothing is invented and no model must be loaded.

A local abstractive backend (``LocalModelSummarizer``) is available for the
live pipeline. It uses a pre-downloaded FLAN-T5 checkpoint, stays fully
offline, and falls back to the extractive path when the model is missing or
produces a weak result.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

from ..config import (
    SUMMARIZER_MODEL_PATH,
    SUMMARIZER_FALLBACK_PATH,
    SUMMARIZER_MAX_INPUT_CHARS,
    SUMMARIZER_MAX_NEW_TOKENS,
    SUMMARIZER_USE_MODEL,
)

logger = logging.getLogger("forgetmenot.audio.summarize")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Unicode-aware words so summaries also work for non-English conversations.
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*")

# Self-introduction patterns. Deliberately conservative: only confident
# name/relationship forms, so we never invent a person from vague speech.
_NAME_PATTERNS = [
    re.compile(
        r"\b(?:my name is|i am|i['\u2019]m|im)\s+(?:a\s+)?(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hey|hi|hello|yo)\s+(?:i\s+am|i['\u2019]m|im)\s+(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthis is\s+(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b", re.IGNORECASE),
]
_ROLE_PATTERNS = [
    re.compile(
        r"\b(?:i\s+am|i['\u2019]m|im)\s+your\s+(?P<role>[A-Za-z'\u2019\-]{2,30})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bmy\s+(?P<role>[A-Za-z'\u2019\-]{2,30})\b", re.IGNORECASE),
    re.compile(r"\b(?P<role>[A-Za-z'\u2019\-]{2,30})\s+of\s+mine\b", re.IGNORECASE),
]
_COMMON_FILLERS = frozenset(
    {
        "a", "an", "and", "hello", "hey", "hi", "i", "me", "my", "no", "ok",
        "okay", "that", "the", "this", "yes", "you", "your",
    }
)


def extract_person_intro(text: str) -> Optional[dict]:
    """Extract a self-introduction (name + relationship/role) from speech text.

    Looks for patterns like:

        "hey im Ada"                    -> name: "Ada"
        "my name is John"               -> name: "John"
        "i am your son"                 -> relationship: "son"
        "im your daughter"              -> relationship: "daughter"
        "hey im Mary, im your wife"     -> name: "Mary", relationship: "wife"

    Returns ``{"name": ..., "relationship": ...}`` with only the fields that
    were confidently found, or ``None`` if nothing was detected.
    """
    if not text:
        return None

    found: dict[str, str] = {}
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        name = m.group("name").strip().strip("'-,. ")
        if not name:
            continue
        if name.lower() in _COMMON_FILLERS:
            continue
        found["name"] = name
        break

    for pat in _ROLE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        role = m.group("role").strip().strip("'-,. ")
        if not role:
            continue
        if role.lower() in _COMMON_FILLERS:
            continue
        if role.lower() == "name":
            continue
        found["relationship"] = role
        break

    return found or None


def join_utterances(texts: list[str]) -> str:
    """Join utterances into one block, keeping sentence boundaries intact.

    Whisper sentences sometimes lack final punctuation; joining them with a
    bare space creates run-on sentences where the summarizer repeats the same
    opening words. Adding a period keeps each utterance a clean candidate.
    """
    parts: list[str] = []
    for text in texts:
        t = (text or "").strip()
        if not t:
            continue
        if t[-1] not in ".!?":
            t += "."
        parts.append(t)
    return " ".join(parts)


class ConversationSummarizer:
    """Extractive key-point summarizer.

    Parameters
    ----------
    max_points_per_person:
        Maximum number of key points to emit per person.
    max_words_per_point:
        Each key point is trimmed to at most this many words ("very small").
    min_words_per_point:
        Sentences shorter than this are not considered informative.
    """

    def __init__(
        self,
        max_points_per_person: int = 4,
        max_words_per_point: int = 14,
        min_words_per_point: int = 4,
    ) -> None:
        self.max_points_per_person = max_points_per_person
        self.max_words_per_point = max_words_per_point
        self.min_words_per_point = min_words_per_point

    def summarize_by_person(
        self, segments_by_person: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Summarize each person's utterances into small key points."""
        result: dict[str, list[str]] = {}
        for key, texts in segments_by_person.items():
            combined = join_utterances(texts)
            if not combined:
                result[key] = []
            else:
                result[key] = self._summarize_text(combined)
        return result

    def summarize_text(self, text: str) -> list[str]:
        """Return key points for a single block of text."""
        return self._summarize_text(text)

    # ------------------------------------------------------------- internals ---

    def _summarize_text(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        candidates = [
            s for s in sentences if self._word_count(s) >= self.min_words_per_point
        ]
        if not candidates:
            candidates = sentences[:1]

        scores = self._score_sentences(candidates)

        chosen: list[tuple[int, str]] = []
        seen_texts: list[str] = []
        for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True):
            if len(chosen) >= self.max_points_per_person:
                break
            sentence = candidates[i]
            if self._is_near_duplicate(sentence, seen_texts):
                # Prefer the more complete wording when two candidates overlap.
                c_words = set(_WORD_RE.findall(sentence.lower()))
                for j, (_, existing) in enumerate(chosen):
                    s_words = set(_WORD_RE.findall(existing.lower()))
                    if (
                        len(c_words) > len(s_words)
                        and len(c_words & s_words) / min(len(c_words), len(s_words)) >= 0.85
                    ):
                        chosen[j] = (i, sentence)
                        seen_texts[j] = sentence
                        break
                continue
            chosen.append((i, sentence))
            seen_texts.append(sentence)

        chosen.sort(key=lambda pair: pair[0])
        points = [self._trim_words(sentence) for _, sentence in chosen]
        return [p for p in points if p]

    def _split_sentences(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
        return [p for p in parts if p]

    def _word_count(self, text: str) -> int:
        return len(_WORD_RE.findall(text))

    def _trim_words(self, text: str) -> str:
        words = _WORD_RE.findall(text)
        if len(words) > self.max_words_per_point:
            words = words[: self.max_words_per_point]
        trimmed = " ".join(words)
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        return trimmed

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        word_freq: dict[str, int] = {}
        for sentence in sentences:
            for word in _WORD_RE.findall(sentence.lower()):
                word_freq[word] = word_freq.get(word, 0) + 1

        if not word_freq:
            return [0.0] * len(sentences)

        max_freq = max(word_freq.values())
        scores: list[float] = []
        n = len(sentences)
        for idx, sentence in enumerate(sentences):
            words = [w.lower() for w in _WORD_RE.findall(sentence)]
            if not words:
                scores.append(0.0)
                continue
            tf = sum(word_freq.get(w, 0) for w in words) / (max_freq * len(words))

            position_bonus = 0.0
            if idx == 0:
                position_bonus += 0.15
            if idx == n - 1:
                position_bonus += 0.10

            length_penalty = 0.15 if len(words) > 25 else 0.0

            scores.append(tf + position_bonus - length_penalty)

        return scores

    @staticmethod
    def _is_near_duplicate(
        candidate: str, seen: list[str], threshold: float = 0.7
    ) -> bool:
        c_words = set(_WORD_RE.findall(candidate.lower()))
        if not c_words:
            return True
        for s in seen:
            s_words = set(_WORD_RE.findall(s.lower()))
            if not s_words:
                continue
            overlap = len(c_words & s_words) / len(c_words | s_words)
            if overlap >= threshold:
                return True
            # Also treat a near-subset as duplicate (overlapping ASR regions
            # often produce a shortened re-transcription of the same words).
            containment = len(c_words & s_words) / min(len(c_words), len(s_words))
            if min(len(c_words), len(s_words)) >= 4 and containment >= 0.85:
                return True
        return False


class LocalModelSummarizer:
    """Offline FLAN-T5 summarizer for the live pipeline.

    Loads a pre-downloaded checkpoint from ``data/models`` on first use (never
    the network). If the model is missing, fails to load, or returns something
    too weak, the extractive summarizer produces the key points instead.
    """

    def __init__(
        self,
        model_path: str = SUMMARIZER_MODEL_PATH,
        fallback_path: str = SUMMARIZER_FALLBACK_PATH,
        max_input_chars: int = SUMMARIZER_MAX_INPUT_CHARS,
        max_new_tokens: int = SUMMARIZER_MAX_NEW_TOKENS,
        max_points: int = 2,
        use_model: bool = SUMMARIZER_USE_MODEL,
    ) -> None:
        self._explicit_path = model_path
        self._fallback_path = fallback_path
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.max_points = max_points
        self.use_model = use_model

        self._extractive = ConversationSummarizer(max_points_per_person=max_points)
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._attempted = False
        self._model_path: Optional[str] = None
        self.model_version = "unloaded"

    def _resolve_path(self) -> Optional[str]:
        for candidate in (self._explicit_path, self._fallback_path):
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def _candidate_paths(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for candidate in (self._explicit_path, self._fallback_path):
            if candidate and candidate not in seen and Path(candidate).exists():
                seen.add(candidate)
                out.append(candidate)
        return out

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._attempted:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._attempted:
                return False
            self._attempted = True
            candidates = self._candidate_paths()
            if not candidates:
                logger.warning(
                    "Summarizer model not found (looked for %s and %s); "
                    "using extractive fallback",
                    self._explicit_path, self._fallback_path,
                )
                self.model_version = "extractive-fallback"
                return False
            for path in candidates:
                try:
                    import torch
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
                    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        path, local_files_only=True
                    )
                    model.eval()
                    self._tokenizer = tok
                    self._model = model
                    self._model_path = path
                    self.model_version = f"flan-t5:{Path(path).name}"
                    logger.info("Summarizer model loaded: %s", path)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not load summarizer model at %s (%s); trying next",
                        path, exc,
                    )
                    self._model = None
            self.model_version = "extractive-fallback"
        return self._model is not None

    def is_loaded(self) -> bool:
        return self._model is not None

    def _prompt(self, text: str) -> str:
        return (
            "Write two short, warm, factual key points about what this "
            "person said. Do not add new information or names.\n"
            f"{text}\nKey points:"
        )

    def summarize(self, text: str) -> list[str]:
        """Return 1-2 small key points for one person's utterances."""
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return []

        if not self.use_model:
            self.model_version = "extractive"
            return self._extractive.summarize_text(text)

        # Bound the input, preferring a sentence boundary.
        if len(text) > self.max_input_chars:
            cut = text[: self.max_input_chars]
            boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            text = cut[: boundary + 1] if boundary > 200 else cut

        if not self._ensure_model() or self._tokenizer is None or self._model is None:
            self.model_version = "extractive-fallback"
            return self._extractive.summarize_text(text)

        try:
            import torch

            enc = self._tokenizer(
                self._prompt(text), return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=2,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.15,
                )
            generated = self._tokenizer.decode(out[0], skip_special_tokens=True)
            points = self._clean_points(generated)
            if self._is_usable(points, text):
                return points
            logger.info("Model summary too weak; using extractive fallback")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Summarization failed: %s", exc)

        self.model_version = "extractive-fallback"
        return self._extractive.summarize_text(text)

    def _clean_points(self, generated: str) -> list[str]:
        text = (generated or "").strip()
        lower = text.lower()
        for prefix in ("key points:", "summary:", "answer:", "output:"):
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
        parts = [p.strip(" -•\t") for p in _SENTENCE_SPLIT.split(text) if p.strip()]
        points: list[str] = []
        for part in parts[: self.max_points]:
            words = _WORD_RE.findall(part)
            if len(words) < 3:
                continue
            points.append(part if part[-1] in ".!?" else part + ".")
        return points

    @staticmethod
    def _is_usable(points: list[str], source: str) -> bool:
        """Reject empty, copied, or clearly malformed generations."""
        if not points:
            return False
        source_words = set(_WORD_RE.findall(source.lower()))
        point_words = set()
        for p in points:
            point_words.update(_WORD_RE.findall(p.lower()))
        if len(point_words) < 4:
            return False
        # A pure copy of the input is not a summary.
        if point_words and point_words <= source_words and len(point_words) > 25:
            return False
        return True
