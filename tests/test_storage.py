from __future__ import annotations

import gc
import time
from pathlib import Path
from uuid import uuid4

from source.chat_storage import ChatSession, ChatStore, chat_title_from_first_user_message
from source.models import EqFilter, Preset
from source.storage import PresetStore


def test_preset_store_can_find_by_name_case_insensitive() -> None:
    path = Path(f"_test_presets_{uuid4().hex}.sqlite3")
    try:
        store = PresetStore(path)
        saved = store.save_new(Preset("Warm", [EqFilter()]))
        found = store.get_preset_by_name("warm")
        assert found is not None
        assert found.id == saved.id
    finally:
        gc.collect()
        for _ in range(10):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05)


def test_chat_store_persists_messages_and_context_state() -> None:
    path = Path(f"_test_chats_{uuid4().hex}.sqlite3")
    try:
        store = ChatStore(path)
        title = chat_title_from_first_user_message("сделай звук максимально мягким и спокойным")
        saved = store.save_new(title, [{"role": "user", "content": "сделай мягче"}], context_full=False)
        saved.messages.append({"role": "assistant", "content": "Смягчил верх."})
        saved.context_full = True
        store.update(saved)

        loaded = store.get_session(saved.id or -1)
        assert loaded is not None
        assert loaded.title.endswith("...")
        assert loaded.context_full is True
        assert loaded.messages[-1] == {"role": "assistant", "content": "Смягчил верх."}
    finally:
        gc.collect()
        for _ in range(10):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05)


def test_chat_session_filters_invalid_messages() -> None:
    session = ChatSession(
        "Chat...",
        [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "  hello  "},
            {"role": "assistant", "content": ""},
        ],
    )
    assert session.sanitized_messages() == [{"role": "user", "content": "hello"}]
