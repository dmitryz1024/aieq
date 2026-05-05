from __future__ import annotations

from source.ai import AiEqualizerService
from source.models import flat_preset


def test_ai_provider_none_does_not_create_fallback_preset(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AI_PROVIDER", "none")
    service = AiEqualizerService()
    result = service.suggest_preset("добавь воздуха", flat_preset())
    assert result.preset is None
    assert result.assistant_message == "Ваш ИИ-агент не подключен"
    assert result.connected is False


def test_new_preset_is_named_new() -> None:
    assert flat_preset().name == "New"
