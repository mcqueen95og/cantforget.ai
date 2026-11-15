"""Conversation summarization: small, gentle key points per speaker/person.

Distills a conversation's transcript segments into a handful of short key
points, grouped by who said what (diarized speaker label, linked to a
recognised face identity where available).

Designed for CPU-only, low-RAM hardware (the i3/8GB target). The default
``ConversationSummarizer`` is extractive and dependency-free beyond the stdlib:
it ranks sentences by term frequency and keeps the most informative ones, so
nothing is invented and no model must be loaded.

Person descriptions are built by ``PersonDescriptionWriter`` from the
cumulative fact ledger (``extract_self_claims``): deterministic, offline,
and never invents content.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..config import (
    DESCRIPTION_MAX_FACTS,
    DESCRIPTION_MAX_CHARS,
)

logger = logging.getLogger("cantforget.audio.summarize")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Unicode-aware words so summaries also work for non-English conversations.
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*")

# Self-introduction patterns. Deliberately conservative: only confident
# name/relationship forms, so we never invent a person from vague speech.
_NAME_PATTERNS = [
    re.compile(
        r"\bmy\s+name(?:\u2019s|'s|\s+is)\s+(?:just\s+)?(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+can\s+)?call\s+me\s+(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hey|hi|hello|yo)\s+(?:i\s+am|i['\u2019]m|im)\s+(?:a\s+|an\s+|the\s+)?(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthis\s+is\s+(?:a\s+|the\s+)?(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+am|i['\u2019]m|im)\s+(?:a\s+|an\s+|the\s+)?(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})\b",
        re.IGNORECASE,
    ),
]
# A name said right after a role: "im your daughter mary" / "im your son, john".
_ROLE_FIRST_PAT = re.compile(
    r"\b(?:i\s+am|i['\u2019]m|im)\s+your\s+(?P<role>[A-Za-z'\u2019\-]{2,30})\s*[,!.]?\s*"
    r"(?P<name>[A-Za-z][A-Za-z'\u2019\-]{1,40})?\b",
    re.IGNORECASE,
)
_ROLE_PATTERNS = [
    re.compile(
        r"\b(?:i\s+am|i['\u2019]m|im)\s+(?:your|the)\s+(?P<role>[A-Za-z'\u2019\-]{2,30})\b",
        re.IGNORECASE,
    ),
]
# Only real relationship/role words are accepted as a tag, so phrases like
# "my day was good" can never become someone's tag.
_RELATIONSHIP_WORDS = frozenset(
    {
        "mother", "mom", "mum", "mama", "father", "dad", "papa", "baba",
        "brother", "sister", "son", "daughter", "wife", "husband", "spouse",
        "grandmother", "grandma", "grandfather", "grandpa", "grandson",
        "granddaughter", "grandchild", "uncle", "aunt", "aunty", "auntie",
        "nephew", "niece", "cousin", "friend", "bestfriend", "girlfriend",
        "boyfriend", "partner", "roommate", "flatmate", "neighbour",
        "neighbor", "colleague", "coworker", "classmate", "teacher",
        "student", "doctor", "nurse", "carer", "caregiver", "assistant",
        "helper", "soninlaw", "daughterinlaw", "brotherinlaw", "sisterinlaw",
        "motherinlaw", "fatherinlaw", "stepbrother", "stepsister",
        "stepfather", "stepmother", "stepson", "stepdaughter", "elder",
        "youngest", "eldest", "twin", "landlord", "neighbour",
    }
)
_COMMON_FILLERS = frozenset(
    {
        "a", "an", "and", "but", "so", "because", "to", "for", "at", "in",
        "on", "it", "its", "is", "am", "are", "was", "be", "been", "the",
        "that", "this", "these", "those", "hello", "hey", "hi", "i", "im",
        "me", "my", "no", "not", "ok", "okay", "yes", "you", "your", "yours",
        "here", "speaking", "sorry", "fine", "good", "great", "okay",
        "tired", "hungry", "happy", "sad", "back", "home", "late", "early",
        "ready", "busy", "free", "sure", "right", "wrong", "new", "old",
        "one", "two", "first", "last", "just", "only", "very", "really",
        "trying", "going", "coming", "looking", "thinking", "calling",
        # Prepositions / question words that otherwise become "names"
        # ("I'm from Canada" must not create a person named "From").
        "from", "with", "about", "near", "by", "of", "off", "out", "up",
        "down", "over", "under", "into", "onto", "how", "what", "why",
        "when", "where", "who", "whom", "which",
        # Common self-states, so "I'm retired" never registers a person.
        "retired", "married", "single", "widow", "widower", "divorced",
        "alone", "sick", "ill", "better", "worse", "cold", "hot", "thirsty",
        "well", "glad", "upset", "angry", "scared", "afraid", "bored",
        "lonely", "confused", "sleepy", "done", "lost", "hurt", "weak",
    }
)


def _clean_capture(raw: str) -> str:
    """Trim and normalise a captured name/role."""
    value = (raw or "").strip().strip("'-,. ").strip()
    if not value:
        return ""
    # Whisper often returns lowercase names ("im mary") - title-case them
    # for display unless the text already has internal capitalisation.
    if value.islower() or value.isupper():
        value = value[:1].upper() + value[1:].lower()
    return value


def extract_person_intro(text: str) -> Optional[dict]:
    """Extract a self-introduction (name + relationship/role) from speech text.

    Looks for patterns like:

        "hey im Ada"                    -> name: "Ada"
        "my name is John"               -> name: "John"
        "my name's John"                -> name: "John"
        "you can call me Ada"           -> name: "Ada"
        "i am your son"                 -> relationship: "son"
        "im your daughter"              -> relationship: "daughter"
        "hey im Mary, im your wife"     -> name: "Mary", relationship: "wife"
        "im your daughter Mary"         -> name: "Mary", relationship: "daughter"

    Returns ``{"name": ..., "relationship": ...}`` with only the fields that
    were confidently found, or ``None`` if nothing was detected.
    """
    if not text:
        return None

    found: dict[str, str] = {}

    # Role-first form: "im your daughter mary".
    m = _ROLE_FIRST_PAT.search(text)
    if m and m.group("role").lower().strip("'-") in _RELATIONSHIP_WORDS:
        role = _clean_capture(m.group("role"))
        if role:
            found["relationship"] = role
            name = _clean_capture(m.group("name") or "")
            if name and name.lower() not in _COMMON_FILLERS:
                found["name"] = name

    if "name" not in found:
        for pat in _NAME_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            name = _clean_capture(m.group("name"))
            if not name:
                continue
            if name.lower() in _COMMON_FILLERS:
                continue
            # Reject contractions captured as names ("let's", "it's",
            # "don't" ...). Real names never end in "'s"/"n't".
            if re.search(r"['’](?:s|t)$", name.lower()):
                continue
            # Skip words that are relationship roles ("im your sister" must
            # not become a person named "Your" or a name-less role match).
            if name.lower() in _RELATIONSHIP_WORDS:
                continue
            # Skip article-consumed descriptions ("I'm a mathematician",
            # "I'm the son of Watkins"): the word after a/an/the is a
            # description, not a name.
            if re.search(
                r"\bI(?:\u2019m|'m|\s+am|im)\s+(?:a\s+|an\s+|the\s+)"
                + re.escape(name) + r"\b",
                text, flags=re.IGNORECASE,
            ):
                continue
            found["name"] = name
            break

    if "relationship" not in found:
        for pat in _ROLE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            role = _clean_capture(m.group("role"))
            if not role:
                continue
            if role.lower() in _COMMON_FILLERS:
                continue
            if role.lower() not in _RELATIONSHIP_WORDS:
                continue
            found["relationship"] = role
            break

    return found or None


# --------------------------------------------------------------------------
# Self-claim extraction: turn what a person says about themselves
# ("I'm a mathematician and the son of Watkins") into short third-person
# facts ("He is a mathematician", "Son of Watkins") for their profile.
# Deterministic, offline, instant — the fallback half of the description
# engine (the model half lives in PersonDescriptionWriter.write_facts).
# --------------------------------------------------------------------------

_PREDICATE_PAT = re.compile(
    r"\bI(?:\u2019m|'m|\s+am|\s+was|im)?\s+(?P<pred>[^.!?\n]{2,140})",
    re.IGNORECASE,
)
_ROLE_OF_PAT = re.compile(
    r"^(?:the\s+|your\s+)?"
    r"(?P<role>son|daughter|wife|husband|father|mother|brother|sister|"
    r"grandson|granddaughter|grandmother|grandfather|grandma|grandpa|"
    r"uncle|aunt|auntie|nephew|niece|cousin|friend|partner|carer|"
    r"neighbour|neighbor)\s+of\s+"
    r"(?P<who>[A-Za-z][A-Za-z'\u2019\-]*(?:\s+[A-Z][A-Za-z'\u2019\-]*)?)",
    re.IGNORECASE,
)
_YOUR_ROLE_PAT = re.compile(
    r"^your\s+(?P<role>[A-Za-z'\u2019\-]{2,30})$", re.IGNORECASE,
)
_ARTICLE_NOUN_PAT = re.compile(
    r"^(?:a|an|the)\s+(?P<noun>[a-z][\w\s'\u2019\-]{1,50})$", re.IGNORECASE,
)
_FROM_PAT = re.compile(r"^from\s+(?P<place>[a-z][\w\s'\u2019\-]{1,30})$", re.IGNORECASE)
_AGE_PAT = re.compile(r"^(?P<age>\d{1,3})\s*(?:years?|yrs?)?\s*(?:old)?$")
_WORK_PAT = re.compile(
    r"\bI\s+(?:am\s+)?work(?:ing|s)?\s+(?:as\s+|at\s+|for\s+)?"
    r"(?P<what>[a-z][\w\s'\u2019\-]{2,50}?)(?=[,.!?;:]|$|\s+and\b)",
    re.IGNORECASE,
)
_LIVE_PAT = re.compile(
    r"\bI\s+live\s+in\s+(?P<where>[a-z][\w\s'\u2019\-]{1,30}?)"
    r"(?=[,.!?;:]|\s+(?:and|but|also)\b|$)",
    re.IGNORECASE,
)
_ENJOY_PAT = re.compile(
    r"\bI\s+(?:really\s+|absolutely\s+|just\s+)?(?P<verb>like|love|enjoy)\s+"
    r"(?P<what>[a-z][\w\s'\u2019\-]{2,40}?)"
    r"(?=[,.!?;:]|\s+(?:and|but|also)\b|$)",
    re.IGNORECASE,
)
_HAVE_PAT = re.compile(
    r"\bI\s+have\s+(?P<what>(?:a|an|the)?\s*[a-z][\w\s'\u2019\-]{1,40}?)"
    r"(?=[,.!?;:]|\s+(?:and|but|also)\b|$)",
    re.IGNORECASE,
)
# "I have X" contents that must not become facts ("I have no idea").
_HAVE_JUNK = frozenset(
    {
        "no idea", "idea", "an idea", "a question", "question", "a doubt",
        "doubt", "fun", "time", "a minute", "a second", "a moment",
        "a go", "to go", "to come", "to do", "to say", "to ask",
        "no clue", "clue", "a clue",
    }
)
# Compound first-person sentences: split at clause boundaries that begin a
# new statement, so "I'm a mathematician and I have dementia" yields two
# facts instead of being swallowed by one predicate.
_CLAUSE_SPLIT = re.compile(
    r"(?=\bI(?:\u2019m|'m|\s+am|\s+have|\s+work|\s+live|\s+like|\s+love|"
    r"\s+enjoy|\s+was|\s+from)\b)",
    re.IGNORECASE,
)
_CLAUSE_LEAD = re.compile(r"^(?:\s*(?:,|and|but|also)\s+)+", re.IGNORECASE)
# First-person tokens that must never appear inside a stored fact.
_FIRST_PERSON = re.compile(
    r"\b(?:i|i'm|im|i\u2019m|am|my|mine|me|we|our|us)\b", re.IGNORECASE,
)
# Second-person / question fragments ("how are you") are never facts.
_SECOND_PERSON = re.compile(
    r"\b(?:you|your|you're|you\u2019re|u|how|what|why|when|where|who)\b",
    re.IGNORECASE,
)
# Fragments starting with these verbs are handled by dedicated patterns
# ("I live in Delhi") or are not facts at all ("I think it's late").
_VERB_START = frozenset(
    {
        "live", "lives", "work", "works", "working", "like", "likes",
        "love", "loves", "enjoy", "enjoys", "have", "has", "had", "want",
        "wants", "think", "know", "feel", "felt", "get", "got", "do",
        "does", "did", "make", "made", "come", "came", "go", "went",
        "going", "need", "see", "saw", "been", "being", "tell", "told",
        "say", "said", "speak", "spoke", "call", "called", "try", "tried",
    }
)
_FACT_SPLIT = re.compile(r",|\band\b|\balso\b", re.IGNORECASE)


def _norm_fact(text: str) -> str:
    return " ".join((text or "").split()).strip().strip(" .,-")


def extract_self_claims(text: str, subject: str = "He") -> list[str]:
    """Extract third-person facts from first-person statements.

    Facts are stored with the person's **name** as the subject (never a
    pronoun), so they stay correct no matter when the person's gender
    becomes known ("Sarah is a mathematician", later rendered as
    "She is a mathematician"). Pass the person's name as ``subject``.

    Examples (subject "Rahul"):
        "Hi, I'm Rahul."                        -> "Rahul is Rahul" (skipped:
                                                   the name field covers it)
        "I'm a mathematician"                   -> "Rahul is a mathematician"
        "I'm a mathematician and the son of
         Watkins"                               -> "Rahul is a mathematician",
                                                   "Son of Watkins"
        "I'm your son"                          -> "Rahul is your son"
        "I work as a teacher"                   -> "Rahul works as a teacher"
        "I live in Delhi" / "I'm from Delhi"    -> "Rahul lives in Delhi" /
                                                   "Rahul is from Delhi"
        "I like chess"                          -> "Rahul likes chess"
        "I'm 70 years old"                      -> "Rahul is 70 years old"

    Returns a list of short fact strings (without bullet markers). Junk
    ("I'm fine", "I'm here") is rejected via the filler word list.
    """
    if not text:
        return []
    facts: list[str] = []
    subj = (subject or "He").strip() or "He"

    def accept(fact: str) -> None:
        fact = _norm_fact(fact)
        if not fact:
            return
        words = _WORD_RE.findall(fact)
        if not 2 <= len(words) <= 14:
            return
        if _FIRST_PERSON.search(fact):
            return
        # Reject pure filler content ("Sarah is fine", "Sarah is here").
        tail = fact
        for lead in (f"{subj} is", f"{subj} are", f"{subj} was"):
            if tail.lower().startswith(lead.lower()):
                tail = tail[len(lead):].strip()
                break
        tail_key = tail.lower().strip(".,'-")
        if not tail_key or tail_key in _COMMON_FILLERS:
            return
        facts.append(fact[0].upper() + fact[1:] if fact[:1].islower() else fact)

    # A stated name is a fact too ("Hi, I'm Rahul") — but skip it when it
    # is the same person's own name (the name field already says that).
    intro = extract_person_intro(text)
    if intro and intro.get("name"):
        if intro["name"].lower() != subj.lower():
            accept(f"{subj} is {intro['name']}")

    # Compound sentences are split at each new first-person statement, so
    # "I'm a mathematician and I have dementia" produces both facts.
    clauses = [
        _CLAUSE_LEAD.sub("", piece).strip()
        for piece in _CLAUSE_SPLIT.split(text)
    ]
    for clause in clauses:
        if not clause:
            continue
        for m in _WORK_PAT.finditer(clause):
            what = _norm_fact(m.group("what"))
            if what and what.lower() not in _COMMON_FILLERS:
                accept(f"{subj} works as {what}")
        for m in _LIVE_PAT.finditer(clause):
            where = _clean_capture(m.group("where"))
            if where:
                accept(f"{subj} lives in {where}")
        for m in _ENJOY_PAT.finditer(clause):
            what = _norm_fact(m.group("what"))
            verb = m.group("verb").lower()
            maps = {"like": "likes", "love": "loves", "enjoy": "enjoys"}
            if what and what.lower() not in _COMMON_FILLERS:
                accept(f"{subj} {maps[verb]} {what}")
        for m in _HAVE_PAT.finditer(clause):
            what = _norm_fact(m.group("what"))
            if not what:
                continue
            what_key = re.sub(r"^(?:a|an|the)\s+", "", what.lower()).strip()
            if (
                what.lower() in _HAVE_JUNK
                or what_key in _HAVE_JUNK
                or what_key in _COMMON_FILLERS
                or what.lower().startswith("to ")
            ):
                continue
            accept(f"{subj} has {what}")

        # "I'm/I am <predicate>" — split the predicate and classify fragments,
        # so "I'm a mathematician and the son of Watkins" yields both facts.
        for m in _PREDICATE_PAT.finditer(clause):
            pred = m.group("pred").strip()
            if not pred:
                continue
            for frag in _FACT_SPLIT.split(pred):
                frag = _norm_fact(frag)
                if not frag:
                    continue
                rm = _ROLE_OF_PAT.match(frag)
                if rm:
                    role = _clean_capture(rm.group("role"))
                    who = _clean_capture(rm.group("who"))
                    if role and who and who.lower() not in _COMMON_FILLERS:
                        accept(f"{role} of {who}")
                    continue
                ym = _YOUR_ROLE_PAT.match(frag)
                if ym:
                    role = ym.group("role").lower().strip("'-")
                    if role in _RELATIONSHIP_WORDS:
                        accept(f"{subj} is your {role}")
                    continue
                am = _AGE_PAT.match(frag)
                if am:
                    age = int(am.group("age"))
                    if 1 <= age <= 120:
                        accept(f"{subj} is {age} years old")
                    continue
                fm = _FROM_PAT.match(frag)
                if fm:
                    place = _clean_capture(fm.group("place"))
                    if place and place.lower() not in _COMMON_FILLERS:
                        accept(f"{subj} is from {place}")
                    continue
                nm = _ARTICLE_NOUN_PAT.match(frag)
                if nm:
                    noun = _norm_fact(nm.group("noun"))
                    if noun and noun.lower() not in _COMMON_FILLERS:
                        accept(f"{subj} is a {noun}")
                    continue
                # Bare fragment: a name ("Rahul"), a state ("retired"), etc.
                words = frag.split()
                first = words[0].lower().strip("'-") if words else ""
                frag_c = _clean_capture(frag)
                if (
                    len(words) <= 3
                    and first not in _VERB_START
                    and frag.lower() not in _COMMON_FILLERS
                    and frag_c
                    and frag_c.lower() not in _COMMON_FILLERS
                    and frag_c.lower() != subj.lower()
                    and not _SECOND_PERSON.search(frag)
                ):
                    accept(f"{subj} is {frag_c}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def render_description(
    facts: list[str],
    name: Optional[str] = None,
    pronoun: Optional[str] = None,
    max_facts: int = 8,
    max_chars: int = 460,
) -> str:
    """Render the fact ledger as bullet lines for ``persons.description``.

    Facts are stored name-first ("Sarah is a mathematician"). When the
    person's gender is known (``pronoun`` = "He"/"She"), the leading name
    is swapped for the pronoun at display time — so earlier facts are
    corrected retroactively once e.g. a "your daughter" tag arrives.
    With no known gender the name is kept, which is always correct.
    """
    lines: list[str] = []
    total = 0
    for f in facts:
        f = _norm_fact(f)
        if not f:
            continue
        if pronoun and name and f.lower().startswith(name.lower() + " "):
            f = f"{pronoun}{f[len(name):]}"
        line = f"- {f}"
        if len(lines) >= max_facts or total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


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
        # Kept for pipeline logging compatibility; summarization is
        # always extractive now.
        self.model_version = "extractive"

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


class PersonDescriptionWriter:
    """Builds a person's cumulative description from their fact ledger.

    Deterministic only: facts come from ``extract_self_claims`` (the claim
    extractor). Nothing is ever invented — a fact enters a person's
    description only if the person actually stated it.
    """

    def _fact_ok(self, fact: str, known: list[str]) -> bool:
        """A fact must be short, third-person, and genuinely new."""
        fact = _norm_fact(fact)
        if not fact:
            return False
        words = _WORD_RE.findall(fact)
        if not 2 <= len(words) <= 14:
            return False
        if _FIRST_PERSON.search(fact):
            return False
        for k in known or []:
            if ConversationSummarizer._is_near_duplicate(fact, [k], 0.6):
                return False
        return True

    def write_facts(
        self,
        name: str,
        utterances: list[str],
        known_facts: Optional[list[str]] = None,
        pronoun: Optional[str] = None,
    ) -> list[str]:
        """Extract new facts a person stated about themselves.

        Facts are returned with the person's NAME as the subject
        ("Sarah is a mathematician"); the pronoun is applied at render
        time. Returns only facts that are not near-duplicates of
        ``known_facts``.
        """
        utterances = [u for u in (utterances or []) if u and u.strip()]
        known = list(known_facts or [])
        found: list[str] = []

        # Deterministic claim extractor — the sole fact source.
        subject = (name or "").strip() or (pronoun or "He")
        for utt in utterances:
            found.extend(extract_self_claims(utt, subject=subject))

        out: list[str] = []
        seen: set[str] = set()
        accepted = list(known)
        for f in found:
            if not self._fact_ok(f, accepted):
                continue
            key = f.lower()
            if key in seen:
                continue
            seen.add(key)
            accepted.append(f)
            out.append(f)
        return out
