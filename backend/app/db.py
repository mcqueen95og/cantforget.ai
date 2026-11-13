"""SQLite persistence for enrolled people, face embeddings, and voice data.

Only what recognition needs is stored: profile text, one small consented
thumbnail, and 128-d SFace embeddings. Raw camera frames are never stored.
Voice embeddings and transcription segments are stored for Phase 3; raw
audio is never persisted.
``ON DELETE CASCADE`` + ``secure_delete`` make deleting a person remove their
biometric rows (and overwrite the freed pages).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("cantforget.database")

EMBEDDING_DIM = 128
VOICE_EMBEDDING_DIM = 512  # 3D-Speaker ERes2Net embedding (audio/diarize.py)

MAX_NAME = 80
MAX_TAG = 40
MAX_DESCRIPTION = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    uid         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    tag         TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    thumbnail   BLOB,
    consent_at  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_uid TEXT NOT NULL REFERENCES persons(uid) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_uid);

-- Phase 3: voice enrollment and conversation data
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT
);
CREATE TABLE IF NOT EXISTS voice_enrollments (
    id         TEXT PRIMARY KEY,
    person_uid TEXT NOT NULL REFERENCES persons(uid) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,      -- serialized float32 speaker embedding
    model_version TEXT NOT NULL DEFAULT 'silero:256d',
    created_at TEXT NOT NULL,
    consent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_enrollments_person ON voice_enrollments(person_uid);

CREATE TABLE IF NOT EXISTS transcription_segments (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id     TEXT NOT NULL,          -- temporary SPEAKER_XX label
    identity_uid   TEXT REFERENCES persons(uid),  -- linked person (if associated)
    text           TEXT NOT NULL,
    start_ms       INTEGER NOT NULL,
    end_ms         INTEGER NOT NULL,
    confidence     REAL,
    language       TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcript_session ON transcription_segments(session_id);
CREATE INDEX IF NOT EXISTS idx_transcript_speaker ON transcription_segments(speaker_id);

CREATE TABLE IF NOT EXISTS speaker_sessions (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id     TEXT NOT NULL,
    identity_uid   TEXT REFERENCES persons(uid),
    first_seen_ms  INTEGER NOT NULL,
    last_seen_ms   INTEGER NOT NULL,
    segment_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_speaker_sessions_session ON speaker_sessions(session_id);

CREATE TABLE IF NOT EXISTS voice_associations (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id     TEXT NOT NULL,
    identity_uid   TEXT NOT NULL REFERENCES persons(uid) ON DELETE CASCADE,
    confidence     REAL NOT NULL,
    method         TEXT NOT NULL DEFAULT 'timestamp_sync',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_associations_speaker ON voice_associations(speaker_id);
CREATE INDEX IF NOT EXISTS idx_voice_associations_identity ON voice_associations(identity_uid);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    identity_uid   TEXT REFERENCES persons(uid) ON DELETE CASCADE,
    speaker_id     TEXT NOT NULL,
    key_points     TEXT NOT NULL,      -- JSON array of short, gentle key points
    summary_text   TEXT DEFAULT '',    -- full structured summary text
    topics         TEXT DEFAULT '[]',  -- JSON array of major discussion topics
    decisions      TEXT DEFAULT '[]',  -- JSON array of explicit decisions
    model_version  TEXT DEFAULT '',    -- model used for generation
    window_id      TEXT REFERENCES conversation_windows(id) ON DELETE SET NULL,
    start_ms       INTEGER DEFAULT 0,  -- live summarizer: span of source audio
    end_ms         INTEGER DEFAULT 0,
    segment_count  INTEGER DEFAULT 0,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_session ON conversation_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_identity ON conversation_summaries(identity_uid);

CREATE TABLE IF NOT EXISTS conversation_windows (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    window_index   INTEGER NOT NULL,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    status         TEXT NOT NULL DEFAULT 'open',   -- open | finalized | processing | completed | failed
    segment_count  INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_windows_session ON conversation_windows(session_id);

CREATE TABLE IF NOT EXISTS extracted_information (
    id                     TEXT PRIMARY KEY,
    session_id             TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    window_id              TEXT REFERENCES conversation_windows(id) ON DELETE SET NULL,
    participant_uid        TEXT REFERENCES persons(uid) ON DELETE SET NULL,
    speaker_id             TEXT NOT NULL,
    category               TEXT NOT NULL,          -- fact | interest | preference | project | commitment | decision
    normalized_info        TEXT NOT NULL,
    original_statement     TEXT NOT NULL,
    statement_start_ms     INTEGER,
    statement_end_ms       INTEGER,
    model_version          TEXT NOT NULL,
    confidence             REAL,
    verification_status    TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | corrected
    corrected_info         TEXT,
    created_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_extracted_info_session ON extracted_information(session_id);
CREATE INDEX IF NOT EXISTS idx_extracted_info_participant ON extracted_information(participant_uid);
CREATE INDEX IF NOT EXISTS idx_extracted_info_status ON extracted_information(verification_status);

CREATE TABLE IF NOT EXISTS participant_memories (
    id             TEXT PRIMARY KEY,
    participant_uid TEXT NOT NULL REFERENCES persons(uid) ON DELETE CASCADE,
    description    TEXT NOT NULL DEFAULT '',
    interests      TEXT NOT NULL DEFAULT '[]',     -- JSON array
    preferences    TEXT NOT NULL DEFAULT '[]',     -- JSON array
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_participant_memories_person ON participant_memories(participant_uid);
"""


class ValidationError(ValueError):
    """Raised for invalid profile input (maps to HTTP 422/400)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_profile(name: str, tag: str = "", description: str = "") -> tuple[str, str, str]:
    name = (name or "").strip()
    tag = (tag or "").strip()
    description = (description or "").strip()
    if not name:
        raise ValidationError("Name is required.")
    if len(name) > MAX_NAME:
        raise ValidationError(f"Name must be at most {MAX_NAME} characters.")
    if len(tag) > MAX_TAG:
        raise ValidationError(f"Tag must be at most {MAX_TAG} characters.")
    if len(description) > MAX_DESCRIPTION:
        raise ValidationError(f"Description must be at most {MAX_DESCRIPTION} characters.")
    return name, tag, description


def new_uid() -> str:
    return "p_" + uuid.uuid4().hex[:10]


def _to_blob(vec: np.ndarray) -> bytes:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embedding must have {EMBEDDING_DIM} values, got {v.shape[0]}")
    return v.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class PeopleDB:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA secure_delete = ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self._migrate_conversation_tables()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate_conversation_tables(self) -> None:
        """Add new columns to existing conversation_summaries table if missing."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        migrations = []
        if "summary_text" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN summary_text TEXT DEFAULT ''")
        if "topics" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN topics TEXT DEFAULT '[]'")
        if "decisions" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN decisions TEXT DEFAULT '[]'")
        if "model_version" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN model_version TEXT DEFAULT ''")
        if "window_id" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN window_id TEXT REFERENCES conversation_windows(id) ON DELETE SET NULL")
        if "start_ms" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN start_ms INTEGER DEFAULT 0")
        if "end_ms" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN end_ms INTEGER DEFAULT 0")
        if "segment_count" not in cols:
            migrations.append("ALTER TABLE conversation_summaries ADD COLUMN segment_count INTEGER DEFAULT 0")
        for stmt in migrations:
            self._conn.execute(stmt)
        if migrations:
            self._conn.commit()
            logger.info("Migrated conversation_summaries: %d column(s) added", len(migrations))

    # --- writes ---

    def ensure_session(self, session_id: str) -> None:
        """Create the conversation session row if it does not exist."""
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
                (session_id, now),
            )
            self._conn.commit()

    def create_person(
        self,
        name: str,
        tag: str,
        description: str,
        embeddings: list[np.ndarray],
        thumbnail: Optional[bytes] = None,
        consent: bool = False,
    ) -> dict:
        if not consent:
            raise ValidationError("Consent is required before registering a face.")
        name, tag, description = clean_profile(name, tag, description)
        if not embeddings:
            raise ValidationError("At least one face sample is required.")
        blobs = [_to_blob(e) for e in embeddings]
        uid, now = new_uid(), _now()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO persons (uid, name, tag, description, thumbnail, consent_at,"
                    " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (uid, name, tag, description, thumbnail, now, now, now),
                )
                self._conn.executemany(
                    "INSERT INTO face_embeddings (person_uid, embedding, created_at) VALUES (?,?,?)",
                    [(uid, b, now) for b in blobs],
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_person(uid)  # type: ignore[return-value]

    def create_person_provisional(
        self,
        name: str,
        tag: str = "",
        description: str = "",
        thumbnail: Optional[bytes] = None,
    ) -> dict:
        """Create a person record, optionally with a face thumbnail.

        Used for people introduced by speech (e.g. "hey im X, im your son"):
        the name and relationship come from the conversation, and the face
        captured by the camera at that moment is stored so the person is
        recognised next time they appear.
        """
        name, tag, description = clean_profile(name, tag, description)
        uid, now = new_uid(), _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO persons (uid, name, tag, description, thumbnail, consent_at,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (uid, name, tag, description, thumbnail, now, now, now),
            )
            self._conn.commit()
        return self.get_person(uid)  # type: ignore[return-value]

    def add_embeddings(self, uid: str, embeddings: list[np.ndarray]) -> int:
        blobs = [_to_blob(e) for e in embeddings]
        now = _now()
        with self._lock:
            if self.get_person(uid) is None:
                raise KeyError(uid)
            self._conn.executemany(
                "INSERT INTO face_embeddings (person_uid, embedding, created_at) VALUES (?,?,?)",
                [(uid, b, now) for b in blobs],
            )
            self._conn.execute("UPDATE persons SET updated_at=? WHERE uid=?", (now, uid))
            self._conn.commit()
        return len(blobs)

    def set_person_thumbnail(self, uid: str, thumbnail: bytes) -> None:
        """Attach/replace the thumbnail for a person."""
        now = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE persons SET thumbnail=?, updated_at=? WHERE uid=?",
                (thumbnail, now, uid),
            )
            self._conn.commit()

    def update_person(self, uid: str, name: str, tag: str, description: str) -> dict:
        name, tag, description = clean_profile(name, tag, description)
        with self._lock:
            cur = self._conn.execute(
                "UPDATE persons SET name=?, tag=?, description=?, updated_at=? WHERE uid=?",
                (name, tag, description, _now(), uid),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                raise KeyError(uid)
        return self.get_person(uid)  # type: ignore[return-value]

    def delete_person(self, uid: str) -> bool:
        """Delete a person and (via cascade) all of their embeddings.

        Transcript rows only reference the person as an annotation, so those
        links are detached instead of cascading (the words said remain).
        """
        with self._lock:
            self._conn.execute(
                "UPDATE transcription_segments SET identity_uid=NULL WHERE identity_uid=?",
                (uid,),
            )
            self._conn.execute(
                "UPDATE speaker_sessions SET identity_uid=NULL WHERE identity_uid=?",
                (uid,),
            )
            cur = self._conn.execute("DELETE FROM persons WHERE uid=?", (uid,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_all_people(self) -> int:
        """Remove every person and all their stored biometric data.

        Transcript segments keep their text; only the person links and the
        person records themselves (faces, voices, thumbnails, memories) are
        removed.
        """
        with self._lock:
            self._conn.execute("UPDATE transcription_segments SET identity_uid=NULL")
            self._conn.execute("UPDATE speaker_sessions SET identity_uid=NULL")
            cur = self._conn.execute("DELETE FROM persons")
            self._conn.commit()
            return cur.rowcount

    # --- reads ---

    @staticmethod
    def _public(row: sqlite3.Row, sample_count: int) -> dict:
        return {
            "uid": row["uid"],
            "name": row["name"],
            "tag": row["tag"],
            "description": row["description"],
            "has_thumbnail": row["thumbnail"] is not None,
            "samples": sample_count,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_person(self, uid: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM persons WHERE uid=?", (uid,)).fetchone()
            if row is None:
                return None
            n = self._conn.execute(
                "SELECT COUNT(*) FROM face_embeddings WHERE person_uid=?", (uid,)
            ).fetchone()[0]
        return self._public(row, n)

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """Return an existing person by exact name (case-insensitive), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM persons WHERE lower(name) = lower(?) LIMIT 1", (name,)
            ).fetchone()
            if row is None:
                return None
            n = self._conn.execute(
                "SELECT COUNT(*) FROM face_embeddings WHERE person_uid=?", (row["uid"],)
            ).fetchone()[0]
        return self._public(row, n)

    def list_people(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM persons ORDER BY created_at, uid").fetchall()
            counts = dict(
                self._conn.execute(
                    "SELECT person_uid, COUNT(*) FROM face_embeddings GROUP BY person_uid"
                ).fetchall()
            )
        return [self._public(r, counts.get(r["uid"], 0)) for r in rows]

    def get_thumbnail(self, uid: str) -> Optional[bytes]:
        with self._lock:
            row = self._conn.execute("SELECT thumbnail FROM persons WHERE uid=?", (uid,)).fetchone()
        return bytes(row["thumbnail"]) if row and row["thumbnail"] is not None else None

    def load_gallery(self) -> dict[str, np.ndarray]:
        """Return {person_uid: (n, 128) float32 array} for matching. Internal only."""
        gallery: dict[str, list[np.ndarray]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_uid, embedding FROM face_embeddings ORDER BY id"
            ).fetchall()
        for r in rows:
            gallery.setdefault(r["person_uid"], []).append(_from_blob(r["embedding"]))
        return {uid: np.vstack(vs) for uid, vs in gallery.items()}

    # ------------------------------------------------------------------ voice ---

    def add_voice_enrollment(
        self,
        person_uid: str,
        embedding: np.ndarray,
        consent_at: str,
        model_version: str = "silero:256d",
    ) -> str:
        """Store a consented voice embedding for a person."""
        eid = "v_" + uuid.uuid4().hex[:12]
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != VOICE_EMBEDDING_DIM:
            raise ValueError(
                f"Voice embedding must have {VOICE_EMBEDDING_DIM} values, got {vec.shape[0]}"
            )
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO voice_enrollments (id, person_uid, embedding, model_version, created_at, consent_at)"
                " VALUES (?,?,?,?,?,?)",
                (eid, person_uid, vec.tobytes(), model_version, now, consent_at),
            )
            self._conn.commit()
        return eid

    def load_voice_gallery(self) -> dict[str, np.ndarray]:
        """Return {person_uid: (n, 512) float32 array} of voice embeddings.

        Embeddings stored with an older/incompatible model are skipped so a
        dimension mismatch can never break matching.
        """
        gallery: dict[str, list[np.ndarray]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_uid, embedding FROM voice_enrollments ORDER BY id"
            ).fetchall()
        for r in rows:
            vec = _from_blob(r["embedding"])
            if vec.shape[0] != VOICE_EMBEDDING_DIM:
                continue
            gallery.setdefault(r["person_uid"], []).append(vec)
        return {uid: np.vstack(vs) for uid, vs in gallery.items()}

    def delete_voice_enrollments(self, person_uid: str) -> int:
        """Delete all voice enrollments for a person. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM voice_enrollments WHERE person_uid=?", (person_uid,)
            )
            self._conn.commit()
            return cur.rowcount

    def add_transcription_segment(
        self,
        session_id: str,
        speaker_id: str,
        text: str,
        start_ms: int,
        end_ms: int,
        confidence: Optional[float] = None,
        language: Optional[str] = None,
        identity_uid: Optional[str] = None,
    ) -> str:
        """Persist a transcription segment."""
        sid = "t_" + uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO transcription_segments (id, session_id, speaker_id, identity_uid, text, start_ms, end_ms, confidence, language, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, session_id, speaker_id, identity_uid, text, start_ms, end_ms, confidence, language, now),
            )
            self._conn.commit()
        return sid

    def link_speaker_to_identity(
        self,
        session_id: str,
        speaker_id: str,
        identity_uid: str,
        confidence: float,
        method: str = "timestamp_sync",
    ) -> str:
        """Record a voice-to-face association."""
        aid = "a_" + uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO voice_associations (id, session_id, speaker_id, identity_uid, confidence, method, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (aid, session_id, speaker_id, identity_uid, confidence, method, now),
            )
            # Also update speaker_sessions to reflect the link (upsert: the
            # row may not exist yet for this speaker).
            cur = self._conn.execute(
                "UPDATE speaker_sessions SET identity_uid=?, last_seen_ms=last_seen_ms WHERE session_id=? AND speaker_id=?",
                (identity_uid, session_id, speaker_id),
            )
            if cur.rowcount == 0:
                self._conn.execute(
                    "INSERT INTO speaker_sessions (id, session_id, speaker_id, identity_uid,"
                    " first_seen_ms, last_seen_ms, segment_count) VALUES (?,?,?,?,?,?,?)",
                    ("ss_" + uuid.uuid4().hex[:12], session_id, speaker_id, identity_uid, 0, 0, 0),
                )
            self._conn.commit()
        return aid

    def get_speaker_identity(self, session_id: str, speaker_id: str) -> Optional[str]:
        """Return the person uid previously linked to this speaker, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT identity_uid FROM speaker_sessions WHERE session_id=? AND speaker_id=?"
                " AND identity_uid IS NOT NULL LIMIT 1",
                (session_id, speaker_id),
            ).fetchone()
        return row["identity_uid"] if row else None

    def get_voice_associations(self, session_id: str) -> list[dict]:
        """Return all voice-to-face associations for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT speaker_id, identity_uid, confidence, method, created_at FROM voice_associations WHERE session_id=?",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_transcript_segments(
        self, session_id: str, limit: int = 100
    ) -> list[dict]:
        """Return recent transcription segments for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT speaker_id, identity_uid, text, start_ms, end_ms, confidence, language, created_at FROM transcription_segments WHERE session_id=? ORDER BY start_ms LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- conversation summaries ---

    # ------------------------------------------------------------------ summaries ---

    def _summary_rows_to_dicts(self, rows) -> list[dict]:
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            for field in ("key_points", "topics", "decisions"):
                try:
                    d[field] = json.loads(d[field] or "[]")
                except Exception:  # noqa: BLE001
                    d[field] = []
            out.append(d)
        return out

    def get_conversation_summaries(self, session_id: str, limit: int = 50) -> list[dict]:
        """Return all summaries for a session, person name joined when available."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT cs.id, cs.session_id, cs.identity_uid, cs.speaker_id, cs.key_points,"
                " cs.summary_text, cs.topics, cs.decisions, cs.model_version,"
                " cs.start_ms, cs.end_ms, cs.segment_count,"
                " cs.created_at, p.name AS person_name"
                " FROM conversation_summaries cs"
                " LEFT JOIN persons p ON cs.identity_uid = p.uid"
                " WHERE cs.session_id = ?"
                " ORDER BY cs.created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return self._summary_rows_to_dicts(rows)

    def get_conversation_summaries_for_person(
        self, session_id: str, identity_uid: str, limit: int = 50
    ) -> list[dict]:
        """Return summaries for a specific person within a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT cs.id, cs.session_id, cs.identity_uid, cs.speaker_id, cs.key_points,"
                " cs.summary_text, cs.topics, cs.decisions, cs.model_version,"
                " cs.start_ms, cs.end_ms, cs.segment_count,"
                " cs.created_at, p.name AS person_name"
                " FROM conversation_summaries cs"
                " LEFT JOIN persons p ON cs.identity_uid = p.uid"
                " WHERE cs.session_id = ? AND cs.identity_uid = ?"
                " ORDER BY cs.created_at DESC LIMIT ?",
                (session_id, identity_uid, limit),
            ).fetchall()
        return self._summary_rows_to_dicts(rows)

    def get_person_conversation_summaries(
        self, person_uid: str, limit: int = 50
    ) -> list[dict]:
        """Return summaries linked to a recognized person, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT cs.id, cs.session_id, cs.identity_uid, cs.speaker_id, cs.key_points,"
                " cs.summary_text, cs.topics, cs.decisions, cs.model_version,"
                " cs.start_ms, cs.end_ms, cs.segment_count,"
                " cs.created_at, p.name AS person_name"
                " FROM conversation_summaries cs"
                " LEFT JOIN persons p ON cs.identity_uid = p.uid"
                " WHERE cs.identity_uid = ?"
                " ORDER BY cs.created_at DESC LIMIT ?",
                (person_uid, limit),
            ).fetchall()
        return self._summary_rows_to_dicts(rows)

    # ------------------------------------------------------------------ windows ---

    def create_conversation_window(
        self, session_id: str, window_index: int, started_at: str,
        window_id: Optional[str] = None,
    ) -> str:
        """Create a new conversation window for a session."""
        wid = window_id or ("cw_" + uuid.uuid4().hex[:12])
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_windows (id, session_id, window_index, started_at, status, segment_count, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (wid, session_id, window_index, started_at, "open", 0, now),
            )
            self._conn.commit()
        return wid

    def finalize_conversation_window(
        self, window_id: str, ended_at: str, segment_count: int
    ) -> bool:
        """Mark a window as finalized (ready for intelligence processing)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conversation_windows SET ended_at=?, status='finalized', segment_count=? WHERE id=? AND status='open'",
                (ended_at, segment_count, window_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_open_windows(self, session_id: str) -> list[dict]:
        """Return open (not yet finalized) windows for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, window_index, started_at, ended_at, status, segment_count, created_at FROM conversation_windows WHERE session_id=? AND status='open' ORDER BY window_index",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_windows(self, session_id: str, limit: int = 50) -> list[dict]:
        """Return all windows for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, window_index, started_at, ended_at, status, segment_count, created_at FROM conversation_windows WHERE session_id=? ORDER BY window_index DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- extended summaries ---

    def add_conversation_summary(
        self,
        session_id: str,
        speaker_id: str,
        key_points: list[str],
        identity_uid: Optional[str] = None,
        summary_text: str = "",
        topics: Optional[list[str]] = None,
        decisions: Optional[list[str]] = None,
        model_version: str = "",
        window_id: Optional[str] = None,
        start_ms: int = 0,
        end_ms: int = 0,
        segment_count: int = 0,
    ) -> str:
        """Persist a structured conversation summary."""
        sid = "cs_" + uuid.uuid4().hex[:12]
        now = _now()
        points_json = json.dumps(key_points, ensure_ascii=False)
        topics_json = json.dumps(topics or [], ensure_ascii=False)
        decisions_json = json.dumps(decisions or [], ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_summaries (id, session_id, identity_uid, speaker_id, key_points, summary_text, topics, decisions, model_version, window_id, start_ms, end_ms, segment_count, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, session_id, identity_uid, speaker_id, points_json, summary_text, topics_json, decisions_json, model_version, window_id, int(start_ms), int(end_ms), int(segment_count), now),
            )
            self._conn.commit()
        return sid

    # ------------------------------------------------------------ extracted info ---

    def add_extracted_information(
        self,
        session_id: str,
        speaker_id: str,
        category: str,
        normalized_info: str,
        original_statement: str,
        model_version: str,
        participant_uid: Optional[str] = None,
        window_id: Optional[str] = None,
        statement_start_ms: Optional[int] = None,
        statement_end_ms: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Persist a candidate extracted-information record (pending approval)."""
        rid = "ei_" + uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO extracted_information (id, session_id, window_id, participant_uid, speaker_id, category, normalized_info, original_statement, statement_start_ms, statement_end_ms, model_version, confidence, verification_status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, session_id, window_id, participant_uid, speaker_id, category, normalized_info, original_statement, statement_start_ms, statement_end_ms, model_version, confidence, "pending", now),
            )
            self._conn.commit()
        return rid

    def get_pending_extracted_information(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """Return extracted-information records awaiting approval."""
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT ei.id, ei.session_id, ei.window_id, ei.participant_uid, ei.speaker_id, ei.category, ei.normalized_info, ei.original_statement, ei.statement_start_ms, ei.statement_end_ms, ei.model_version, ei.confidence, ei.verification_status, ei.corrected_info, ei.created_at, p.name AS person_name FROM extracted_information ei LEFT JOIN persons p ON ei.participant_uid = p.uid WHERE ei.session_id=? AND ei.verification_status='pending' ORDER BY ei.created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT ei.id, ei.session_id, ei.window_id, ei.participant_uid, ei.speaker_id, ei.category, ei.normalized_info, ei.original_statement, ei.statement_start_ms, ei.statement_end_ms, ei.model_version, ei.confidence, ei.verification_status, ei.corrected_info, ei.created_at, p.name AS person_name FROM extracted_information ei LEFT JOIN persons p ON ei.participant_uid = p.uid WHERE ei.verification_status='pending' ORDER BY ei.created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_extracted_information(
        self, session_id: str, status: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """Return extracted-information records for a session, optionally filtered by status."""
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT ei.id, ei.session_id, ei.window_id, ei.participant_uid, ei.speaker_id, ei.category, ei.normalized_info, ei.original_statement, ei.statement_start_ms, ei.statement_end_ms, ei.model_version, ei.confidence, ei.verification_status, ei.corrected_info, ei.created_at, p.name AS person_name FROM extracted_information ei LEFT JOIN persons p ON ei.participant_uid = p.uid WHERE ei.session_id=? AND ei.verification_status=? ORDER BY ei.created_at DESC LIMIT ?",
                    (session_id, status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT ei.id, ei.session_id, ei.window_id, ei.participant_uid, ei.speaker_id, ei.category, ei.normalized_info, ei.original_statement, ei.statement_start_ms, ei.statement_end_ms, ei.model_version, ei.confidence, ei.verification_status, ei.corrected_info, ei.created_at, p.name AS person_name FROM extracted_information ei LEFT JOIN persons p ON ei.participant_uid = p.uid WHERE ei.session_id=? ORDER BY ei.created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def approve_extracted_information(
        self, record_id: str, corrected_info: Optional[str] = None
    ) -> bool:
        """Approve a candidate extracted-information record.

        When approved, the normalized info is also merged into the
        participant's memory record (description / interests / preferences)
        so the conversation context is preserved with the person.
        """
        now = _now()
        with self._lock:
            # Fetch the record before updating so we know who it belongs to
            # and what category it is.
            row = self._conn.execute(
                "SELECT participant_uid, category, normalized_info FROM extracted_information WHERE id=? AND verification_status='pending'",
                (record_id,),
            ).fetchone()
            if row is None:
                return False
            participant_uid = row["participant_uid"]
            category = row["category"]
            normalized = row["normalized_info"]

            if corrected_info is not None:
                self._conn.execute(
                    "UPDATE extracted_information SET verification_status='approved', corrected_info=?, created_at=? WHERE id=? AND verification_status='pending'",
                    (corrected_info, now, record_id),
                )
                # Use the corrected value for memory if provided.
                normalized = corrected_info
            else:
                self._conn.execute(
                    "UPDATE extracted_information SET verification_status='approved', created_at=? WHERE id=? AND verification_status='pending'",
                    (now, record_id),
                )
            self._conn.commit()

        # Merge into participant memory (outside the row lock to keep it short).
        if participant_uid:
            self._merge_approved_info_into_memory(
                participant_uid, category, normalized
            )
        return True

    def _merge_approved_info_into_memory(
        self, participant_uid: str, category: str, normalized_info: str
    ) -> None:
        """Fold an approved extracted fact into a participant's memory record."""
        if not normalized_info:
            return
        mem = self.get_participant_memories(participant_uid)
        description = ""
        interests: list[str] = []
        preferences: list[str] = []
        if mem:
            description = mem.get("description", "") or ""
            interests = list(mem.get("interests") or [])
            preferences = list(mem.get("preferences") or [])

        lower = normalized_info.lower().strip()

        if category in ("interest", "preference"):
            # Avoid duplicates.
            target = interests if category == "interest" else preferences
            if lower and lower not in [i.lower() for i in target]:
                target.append(lower)
        else:
            # fact / commitment / decision / project -> append to description.
            if description:
                description += ". " + normalized_info
            else:
                description = normalized_info

        self.upsert_participant_memories(
            participant_uid,
            description=description,
            interests=interests,
            preferences=preferences,
        )

    def reject_extracted_information(self, record_id: str) -> bool:
        """Reject (discard) a pending extracted-information record."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE extracted_information SET verification_status='rejected' WHERE id=? AND verification_status='pending'",
                (record_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ participant memories ---

    def get_participant_memories(self, participant_uid: str) -> Optional[dict]:
        """Return the approved memory record for a participant."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, participant_uid, description, interests, preferences, updated_at FROM participant_memories WHERE participant_uid=?",
                (participant_uid,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["interests"] = json.loads(d["interests"] or "[]")
        except Exception:  # noqa: BLE001
            d["interests"] = []
        try:
            d["preferences"] = json.loads(d["preferences"] or "[]")
        except Exception:  # noqa: BLE001
            d["preferences"] = []
        return d

    def upsert_participant_memories(
        self,
        participant_uid: str,
        description: str = "",
        interests: Optional[list[str]] = None,
        preferences: Optional[list[str]] = None,
    ) -> str:
        """Create or update a participant's approved memory record."""
        rid = "pm_" + uuid.uuid4().hex[:12]
        now = _now()
        interests_json = json.dumps(interests or [], ensure_ascii=False)
        preferences_json = json.dumps(preferences or [], ensure_ascii=False)
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM participant_memories WHERE participant_uid=?", (participant_uid,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE participant_memories SET description=?, interests=?, preferences=?, updated_at=? WHERE participant_uid=?",
                    (description, interests_json, preferences_json, now, participant_uid),
                )
                rid = existing["id"]
            else:
                self._conn.execute(
                    "INSERT INTO participant_memories (id, participant_uid, description, interests, preferences, updated_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (rid, participant_uid, description, interests_json, preferences_json, now),
                )
            self._conn.commit()
        return rid

    def delete_participant_memories(self, participant_uid: str) -> bool:
        """Delete a participant's memory record."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM participant_memories WHERE participant_uid=?", (participant_uid,)
            )
            self._conn.commit()
            return cur.rowcount > 0
