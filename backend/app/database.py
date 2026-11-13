"""SQLite database schema, connection, and CRUD for Phase 2.

Single-file database at DATA_DIR / db.sqlite. Tables:

  persons          - enrolled familiar people (name, relationship, ...)
  face_embeddings  - one or more 128-d SFace embeddings per person
  sessions         - conversation/camera sessions
  face_tracks      - temporary face tracks within a session, linked to persons
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import DB_PATH

logger = logging.getLogger("forgetmenot.database")

_DB_PATH = Path(DB_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on startup."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                relationship TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id            TEXT PRIMARY KEY,
                person_id     TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                embedding     BLOB NOT NULL,
                model_version TEXT NOT NULL DEFAULT 'sface:128d',
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS face_tracks (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                track_label  TEXT NOT NULL,
                person_id    TEXT REFERENCES persons(id) ON DELETE CASCADE,
                started_at   TEXT NOT NULL,
                ended_at     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
                ON face_embeddings(person_id);
            CREATE INDEX IF NOT EXISTS idx_face_tracks_person
                ON face_tracks(person_id);
            """
        )
        # Migration: add model_version column if it doesn't exist (existing DBs)
        _migrate_embeddings_table(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_embeddings_table(conn: sqlite3.Connection) -> None:
    """Add model_version column to face_embeddings if missing."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(face_embeddings)").fetchall()]
    if "model_version" not in cols:
        conn.execute(
            "ALTER TABLE face_embeddings ADD COLUMN model_version TEXT NOT NULL DEFAULT 'sface:128d'"
        )
        logger.info("Migrated face_embeddings: added model_version column")


# ---------------- Persons ----------------

def list_persons() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, relationship, created_at, updated_at FROM persons ORDER BY name"
        ).fetchall()
        # Count embeddings per person and collect embedding IDs
        out = []
        for r in rows:
            embed_rows = conn.execute(
                "SELECT id FROM face_embeddings WHERE person_id = ? ORDER BY created_at",
                (r["id"],),
            ).fetchall()
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "relationship": r["relationship"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "embedding_count": len(embed_rows),
                    "embedding_ids": [er["id"] for er in embed_rows],
                }
            )
        return out
    finally:
        conn.close()


def get_person(person_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, name, relationship, created_at, updated_at FROM persons WHERE id = ?",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def create_person(name: str, relationship: Optional[str] = None) -> dict:
    pid = uuid.uuid4().hex
    now = _now_iso()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO persons (id, name, relationship, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (pid, name, relationship, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": pid, "name": name, "relationship": relationship}


def update_person(
    person_id: str, name: Optional[str] = None, relationship: Optional[str] = None
) -> bool:
    conn = get_connection()
    try:
        fields = []
        params: list = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if relationship is not None:
            fields.append("relationship = ?")
            params.append(relationship)
        if not fields:
            return False
        fields.append("updated_at = ?")
        params.append(_now_iso())
        params.append(person_id)
        cur = conn.execute(
            f"UPDATE persons SET {', '.join(fields)} WHERE id = ?", params
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_person(person_id: str) -> bool:
    conn = get_connection()
    try:
        # Delete face_tracks referencing this person first.
        # face_tracks.person_id has NO ON DELETE CASCADE (SQLite limitation:
        # FK clauses can't be altered on existing tables), so we must
        # remove dependent tracks manually before deleting the person.
        conn.execute(
            "DELETE FROM face_tracks WHERE person_id = ?", (person_id,)
        )
        # face_embeddings has ON DELETE CASCADE, so they are removed
        # automatically when the person row is deleted.
        cur = conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------------- Face embeddings ----------------

def add_embedding(person_id: str, embedding: bytes) -> str:
    """Add embedding with default model version (backward compatibility)."""
    return add_embedding_with_version(person_id, embedding, "sface:128d")


def add_embedding_with_version(
    person_id: str, embedding: bytes, model_version: str
) -> str:
    eid = uuid.uuid4().hex
    now = _now_iso()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO face_embeddings (id, person_id, embedding, model_version, created_at) VALUES (?, ?, ?, ?, ?)",
            (eid, person_id, embedding, model_version, now),
        )
        conn.commit()
    finally:
        conn.close()
    return eid


def list_embeddings(person_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, created_at FROM face_embeddings WHERE person_id = ? ORDER BY created_at",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_embedding(embedding_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM face_embeddings WHERE id = ?", (embedding_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_all_embeddings() -> list[tuple[str, bytes, str]]:
    """Return list of (person_id, embedding_bytes, model_version) for the gallery."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT person_id, embedding, model_version FROM face_embeddings"
        ).fetchall()
        return [(r["person_id"], r["embedding"], r["model_version"]) for r in rows]
    finally:
        conn.close()


def embedding_count_for_person(person_id: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM face_embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        return row["c"]
    finally:
        conn.close()


# ---------------- Sessions / tracks ----------------

def create_session() -> str:
    sid = uuid.uuid4().hex
    now = _now_iso()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)", (sid, now)
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def end_session(session_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_now_iso(), session_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def upsert_face_track(
    track_id: str,
    session_id: str,
    track_label: str,
    person_id: Optional[str],
) -> None:
    """Insert or update a face track record."""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM face_tracks WHERE id = ?", (track_id,)
        ).fetchone()
        now = _now_iso()
        if existing:
            conn.execute(
                "UPDATE face_tracks SET person_id = ? WHERE id = ?",
                (person_id, track_id),
            )
        else:
            conn.execute(
                "INSERT INTO face_tracks (id, session_id, track_label, person_id, started_at) VALUES (?, ?, ?, ?, ?)",
                (track_id, session_id, track_label, person_id, now),
            )
        conn.commit()
    finally:
        conn.close()


def end_face_track(track_id: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE face_tracks SET ended_at = ? WHERE id = ?",
            (_now_iso(), track_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
