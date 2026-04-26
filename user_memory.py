from __future__ import annotations

"""Local persistence helpers for the future course project.

Why we store this now:
- Lecture 12 needs session and user tracking for Langfuse.
- The later support project will need returning-user context and ticket history.

Privacy choice:
- We deliberately DO NOT identify people by IP or MAC address.
- IP can change and multiple users can share one network.
- MAC is sensitive device data and should not be collected casually.
- Instead, we create a pseudonymous local user id and persist it on the device.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "support_memory.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    """Create the minimal tables we want as a strong base for the course project."""
    with _get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                preferred_language TEXT,
                profile_summary TEXT,
                consent_scope TEXT DEFAULT 'local_device'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                topic_summary TEXT,
                resolution_status TEXT DEFAULT 'open',
                escalated INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )


def get_or_create_active_user_id() -> str:
    """Return a stable pseudonymous user id stored only on the local machine."""
    ensure_schema()
    with _get_connection() as conn:
        row = conn.execute("SELECT user_id FROM users ORDER BY created_at LIMIT 1").fetchone()
        if row:
            user_id = str(row["user_id"])
            conn.execute(
                "UPDATE users SET updated_at = ? WHERE user_id = ?",
                (_utc_now(), user_id),
            )
            return user_id

        user_id = f"local_user_{uuid4().hex[:12]}"
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO users (user_id, created_at, updated_at, preferred_language, profile_summary, consent_scope)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, now, now, "uk", "Local returning user profile", "local_device"),
        )
        return user_id


def start_new_session(user_id: str) -> str:
    """Create a new session row and return the session id."""
    ensure_schema()
    session_id = f"support_session_{uuid4().hex[:12]}"
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, started_at) VALUES (?, ?, ?)",
            (session_id, user_id, _utc_now()),
        )
    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    """Persist one chat message locally for future follow-up and support history."""
    if not str(content or "").strip():
        return
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, str(content), _utc_now()),
        )


def finish_session(session_id: str, topic_summary: str = "", resolution_status: str = "open", escalated: bool = False) -> None:
    """Mark session outcome so later analytics can count solved vs escalated cases."""
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = ?, topic_summary = ?, resolution_status = ?, escalated = ?
            WHERE session_id = ?
            """,
            (_utc_now(), topic_summary[:500], resolution_status, 1 if escalated else 0, session_id),
        )
