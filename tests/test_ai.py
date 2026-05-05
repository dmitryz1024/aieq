from __future__ import annotations

import numpy as np
from pathlib import Path

from source.ai import (
    AI_SCHEMA,
    DEFAULT_LLAMA_N_CTX,
    MAX_DEVICE_POINTS,
    MAX_DEVICE_RAW_LINES,
    AiEqualizerService,
    sanitize_ai_preset_name,
)
from source.curves import FrequencyCurve
from source.models import EqFilter, Preset, flat_preset


def test_ai_provider_none_does_not_create_fallback_preset(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AI_PROVIDER", "none")
    service = AiEqualizerService()
    result = service.suggest_preset("добавь воздуха", flat_preset())
    assert result.preset is None
    assert result.assistant_message == "Ваш ИИ-агент не подключен"
    assert result.connected is False


def test_llama_cpp_without_model_is_not_connected(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_AI_PROVIDER", "llama_cpp")
    monkeypatch.setenv("AIEQ_LLAMA_MODEL_PATH", "missing-test-model.gguf")
    service = AiEqualizerService()
    result = service.suggest_preset("сделай звук ярче", flat_preset())
    assert result.preset is None
    assert result.assistant_message == "Ваш ИИ-агент не подключен"
    assert result.connected is False


def test_default_llama_context_is_larger(monkeypatch) -> None:
    monkeypatch.delenv("AIEQ_LLAMA_N_CTX", raising=False)
    service = AiEqualizerService()
    assert service.llama_n_ctx == DEFAULT_LLAMA_N_CTX


def test_new_preset_is_named_new() -> None:
    assert flat_preset().name == "New"


def test_ai_generated_name_cannot_use_new_prefix() -> None:
    assert sanitize_ai_preset_name("New V Shape", fallback="Media V Shape") == "Media V Shape"
    assert sanitize_ai_preset_name("Warm Vocal Air Wide", fallback="AI Preset") == "Warm Vocal Air"


def test_ai_schema_does_not_ask_model_for_preset_name() -> None:
    assert "name" not in AI_SCHEMA["properties"]
    assert "name" not in AI_SCHEMA["required"]


def test_device_curve_context_includes_raw_txt() -> None:
    path = Path(__file__).parent / "fixtures" / "device_curve.txt"
    curve = FrequencyCurve("Device", np.array([20.0, 1000.0]), np.array([-1.5, 0.0]), path)
    context = AiEqualizerService()._serialize_device_curve(curve)
    assert context["name"] == "Device"
    assert "20 -1.5" in context["raw_txt"]
    assert context["points"][0] == {"freq": 20.0, "db": -1.5}


def test_device_curve_context_is_compact_for_large_curves() -> None:
    freqs = np.geomspace(20, 20000, 480)
    values = np.linspace(-4.0, 4.0, 480)
    curve = FrequencyCurve("Large Device", freqs, values)
    context = AiEqualizerService()._serialize_device_curve(curve)
    assert len(context["points"]) == MAX_DEVICE_POINTS

    raw_text = "\n".join(f"{index} {index / 10}" for index in range(480))
    sampled, was_sampled = AiEqualizerService._sample_raw_curve_text(raw_text)
    assert was_sampled is True
    assert len(sampled.splitlines()) == MAX_DEVICE_RAW_LINES


def test_gpu_layers_fall_back_to_cpu_when_offload_is_unavailable(monkeypatch) -> None:
    service = AiEqualizerService()
    monkeypatch.setattr(service, "_llama_supports_gpu_offload", lambda: False)
    assert service._effective_gpu_layers(-1) == 0
    assert service._effective_gpu_layers(24) == 0


def test_cpu_gpu_layers_stay_cpu_even_when_offload_is_available(monkeypatch) -> None:
    service = AiEqualizerService()
    monkeypatch.setattr(service, "_llama_supports_gpu_offload", lambda: True)
    assert service._effective_gpu_layers(0) == 0
    assert service._effective_gpu_layers(-1) == -1


def test_clear_context_unloads_llama_instance() -> None:
    class FakeLlama:
        def __init__(self) -> None:
            self.reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    fake = FakeLlama()
    service = AiEqualizerService()
    service._llama = fake
    service._llama_signature = ("model", 8192, 7, 0, 512)
    service.clear_context()
    assert fake.reset_called is True
    assert service._llama is None
    assert service._llama_signature is None


def test_mentioned_saved_preset_is_prioritized_in_compact_context() -> None:
    presets = [
        Preset("Warm", [EqFilter("peaking", 1000, 1, 1)]),
        Preset("AIEQ 2026-05-06 | 01-59-52", [EqFilter("peaking", 2000, 1, 2)]),
    ]
    serialized = AiEqualizerService()._serialize_saved_presets(
        presets,
        user_text='Хочу доработать пресет "AIEQ 2026-05-06 | 01-59-52"',
        compact=True,
    )
    assert serialized[0]["name"] == "AIEQ 2026-05-06 | 01-59-52"


def test_context_limit_errors_are_detected() -> None:
    assert AiEqualizerService._is_context_limit_error(RuntimeError("Requested tokens exceed context window"))
