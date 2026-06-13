from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import utc_now_iso

CHAT_TITLE_MAX_CHARS = 36


def default_chat_db_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "AIEQ" / "chats.sqlite3"


def chat_title_from_first_user_message(text: str, *, max_chars: int = CHAT_TITLE_MAX_CHARS) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        clean = "Новый чат"
    if len(clean) > max_chars:
        clean = clean[:max_chars].rstrip()
    return f"{clean}..."


@dataclass(slots=True)
class ChatSession:
    title: str
    messages: list[dict[str, str]] = field(default_factory=list)
    id: int | None = None
    context_full: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def sanitized_messages(self) -> list[dict[str, str]]:
        clean: list[dict[str, str]] = []
        for message in self.messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                clean.append({"role": role, "content": content})
        return clean


class ChatStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_chat_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    context_full INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC)")

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, messages, context_full, created_at, updated_at FROM chats ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        sessions: list[ChatSession] = []
        for row in rows:
            session = self._session_from_row(row)
            if session is not None:
                sessions.append(session)
        return sessions

    def get_session(self, chat_id: int) -> ChatSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, messages, context_full, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
        return None if row is None else self._session_from_row(row)

    def save_new(self, title: str, messages: list[dict[str, str]] | None = None, *, context_full: bool = False) -> ChatSession:
        now = utc_now_iso()
        session = ChatSession(
            title=title.strip() or chat_title_from_first_user_message(""),
            messages=messages or [],
            context_full=context_full,
            created_at=now,
            updated_at=now,
        )
        data = json.dumps(session.sanitized_messages(), ensure_ascii=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chats(title, messages, context_full, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session.title, data, int(session.context_full), session.created_at, session.updated_at),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an id for the saved chat.")
            session.id = int(cursor.lastrowid)
        return session

    def update(self, session: ChatSession) -> ChatSession:
        if session.id is None:
            return self.save_new(session.title, session.messages, context_full=session.context_full)
        session.messages = session.sanitized_messages()
        session.updated_at = utc_now_iso()
        data = json.dumps(session.messages, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "UPDATE chats SET title = ?, messages = ?, context_full = ?, updated_at = ? WHERE id = ?",
                (session.title, data, int(session.context_full), session.updated_at, session.id),
            )
        return session

    def delete(self, chat_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> ChatSession | None:
        try:
            raw_messages: Any = json.loads(row["messages"])
        except (TypeError, json.JSONDecodeError):
            raw_messages = []
        if not isinstance(raw_messages, list):
            raw_messages = []
        session = ChatSession(
            id=int(row["id"]),
            title=str(row["title"]),
            messages=[message for message in raw_messages if isinstance(message, dict)],
            context_full=bool(row["context_full"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        session.messages = session.sanitized_messages()
        return session
