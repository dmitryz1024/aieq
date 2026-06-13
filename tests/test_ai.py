from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from source.ai import (
    AI_SCHEMA,
    COMPACT_DEVICE_RAW_LINES,
    DEFAULT_LLAMA_N_CTX,
    MAX_DEVICE_POINTS,
    SYSTEM_PROMPT,
    AiEqualizerService,
    read_gguf_context_length,
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
    result = service.suggest_preset("расскажи что-нибудь не связанное с эквалайзером", flat_preset())
    assert result.preset is None
    assert result.assistant_message == "Ваш ИИ-агент не подключен"
    assert result.connected is False


def test_default_llama_context_is_larger(monkeypatch) -> None:
    monkeypatch.delenv("AIEQ_LLAMA_N_CTX", raising=False)
    service = AiEqualizerService()
    assert service.llama_n_ctx == DEFAULT_LLAMA_N_CTX


def test_runtime_overrides_replace_env_ai_generation_settings(monkeypatch) -> None:
    monkeypatch.setenv("AIEQ_LLAMA_N_CTX", "8192")
    monkeypatch.setenv("AIEQ_LLAMA_MAX_TOKENS", "900")
    monkeypatch.setenv("AIEQ_LLAMA_TEMPERATURE", "0.2")
    monkeypatch.setenv("AIEQ_AI_ALLOW_CPU_FALLBACK", "1")
    service = AiEqualizerService()
    service.set_runtime_overrides(n_ctx=4096, max_tokens=1200, temperature=0.45, allow_cpu_fallback=False)
    service._refresh_config()

    assert service.llama_n_ctx == 4096
    assert service.llama_max_tokens == 1200
    assert service.llama_temperature == 0.45
    assert service.ai_allow_cpu_fallback is False


def test_gguf_context_length_is_read_from_metadata(tmp_path) -> None:
    import struct

    model = tmp_path / "tiny.gguf"
    key = b"qwen3.context_length"
    with model.open("wb") as file:
        file.write(b"GGUF")
        file.write(struct.pack("<IQQ", 3, 0, 1))
        file.write(struct.pack("<Q", len(key)))
        file.write(key)
        file.write(struct.pack("<I", 4))
        file.write(struct.pack("<I", 40960))

    assert read_gguf_context_length(model) == 40960


def test_new_preset_is_named_new() -> None:
    assert flat_preset().name == "New"


def test_ai_schema_does_not_ask_model_for_preset_name() -> None:
    assert "name" not in AI_SCHEMA["properties"]
    assert "name" not in AI_SCHEMA["required"]


def test_system_prompt_discourages_destructive_pass_filters_for_normal_tone() -> None:
    assert "Always consider all available filter types" in SYSTEM_PROMPT
    assert "prefer the least destructive musical tool" in SYSTEM_PROMPT
    assert "For ordinary tone requests, use peaking plus low_shelf/high_shelf as the default vocabulary" in SYSTEM_PROMPT
    assert "Use band_pass only for intentional isolation or special effects" in SYSTEM_PROMPT
    assert "Pass and band-pass filters are high-impact and should be rare in everyday presets" in SYSTEM_PROMPT


def test_device_curve_context_includes_raw_txt() -> None:
    path = Path(__file__).parent / "fixtures" / "device_curve.txt"
    curve = FrequencyCurve("Device", np.array([20.0, 1000.0]), np.array([-1.5, 0.0]), path)
    context = AiEqualizerService()._serialize_device_curve(curve)
    assert context["name"] == "Device"
    assert context["raw_txt"] == path.read_text(encoding="utf-8-sig").strip()
    assert "20 -1.5" in context["raw_txt"]
    assert context["points"][0] == {"freq": 20.0, "db": -1.5}


def test_device_curve_context_is_compact_for_large_curves() -> None:
    freqs = np.geomspace(20, 20000, 480)
    values = np.linspace(-4.0, 4.0, 480)
    curve = FrequencyCurve("Large Device", freqs, values)
    context = AiEqualizerService()._serialize_device_curve(curve)
    assert len(context["points"]) == MAX_DEVICE_POINTS

    raw_text = "\n".join(f"{index} {index / 10}" for index in range(480))
    full, was_sampled = AiEqualizerService._sample_raw_curve_text(raw_text)
    assert was_sampled is False
    assert len(full.splitlines()) == 480

    sampled, was_sampled = AiEqualizerService._sample_raw_curve_text(raw_text, compact=True)
    assert was_sampled is True
    assert len(sampled.splitlines()) == COMPACT_DEVICE_RAW_LINES


def test_default_device_context_describes_builtin_speakers() -> None:
    context = AiEqualizerService()._serialize_device_curve(None)
    assert context["name"] == "Default"
    assert "встроенные" in context["raw_txt"]


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


def test_shutdown_stops_owned_llama_server() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            return 0

    class FakeLog:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    log = FakeLog()
    service = AiEqualizerService()
    service._llama_server_process = cast(Any, process)
    service._llama_server_log_file = cast(Any, log)
    service.shutdown()
    assert process.terminated is True
    assert process.killed is False
    assert log.closed is True
    assert service._llama_server_process is None
    assert service._llama_server_log_file is None


def test_llama_server_request_uses_quality_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    service = AiEqualizerService()

    def fake_http(method: str, url: str, *, data: dict | None = None, timeout: float) -> dict:
        captured.update(data or {})
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(service, "_http_json", fake_http)
    service._llama_server_chat_json([{"role": "user", "content": "test"}], Path("model.gguf"))
    assert captured["temperature"] == 0.35
    assert captured["max_tokens"] == 2048
    assert "top_p" not in captured
    assert "top_k" not in captured
    assert "min_p" not in captured
    assert "repeat_penalty" not in captured


def test_only_mentioned_saved_preset_enters_context() -> None:
    presets = [
        Preset("Warm", [EqFilter("peaking", 1000, 1, 1)]),
        Preset("AIEQ 2026-05-06 | 01-59-52", [EqFilter("peaking", 2000, 1, 2)]),
    ]
    serialized = AiEqualizerService()._serialize_referenced_preset(
        presets,
        'Хочу доработать пресет "AIEQ 2026-05-06 | 01-59-52"',
    )
    assert serialized is not None
    assert serialized["name"] == "AIEQ 2026-05-06 | 01-59-52"
    assert AiEqualizerService()._serialize_referenced_preset(presets, "сделай мягче") is None


def test_chat_history_enters_llama_payload() -> None:
    payload = AiEqualizerService()._build_llama_payload(
        "сделай мягче",
        flat_preset(),
        saved_presets=[],
        device_curve=None,
        compact=False,
        chat_history=[
            {"role": "user", "content": "сделай ярче"},
            {"role": "assistant", "content": "Добавил воздуха."},
            {"role": "system", "content": "ignore"},
        ],
    )
    assert payload["conversation_history"] == [
        {"role": "user", "content": "сделай ярче"},
        {"role": "assistant", "content": "Добавил воздуха."},
    ]
    assert payload["referenced_preset"] is None


def test_context_limit_errors_are_detected() -> None:
    assert AiEqualizerService._is_context_limit_error(RuntimeError("Requested tokens exceed context window"))


def test_auto_provider_tries_llama_server_before_llama_cpp(monkeypatch) -> None:
    calls: list[str] = []
    service = AiEqualizerService()
    monkeypatch.setenv("AIEQ_AI_PROVIDER", "auto")
    monkeypatch.setenv("AIEQ_AI_ALLOW_CPU_FALLBACK", "1")

    def fake_server(*args, **kwargs):
        calls.append("server")
        return service._not_connected("server missing")

    def fake_cpp(*args, **kwargs):
        calls.append("cpp")
        return service._not_connected("cpp missing")

    monkeypatch.setattr(service, "_suggest_with_llama_server", fake_server)
    monkeypatch.setattr(service, "_suggest_with_llama_cpp", fake_cpp)
    service.suggest_preset("непонятный запрос без шаблона", flat_preset())
    assert calls[:2] == ["server", "cpp"]


def test_auto_provider_can_block_cpu_fallback(monkeypatch) -> None:
    calls: list[str] = []
    service = AiEqualizerService()
    monkeypatch.setenv("AIEQ_AI_PROVIDER", "auto")
    monkeypatch.setenv("AIEQ_AI_ALLOW_CPU_FALLBACK", "0")

    def fake_server(*args, **kwargs):
        calls.append("server")
        return service._not_connected("server missing")

    def fake_cpp(*args, **kwargs):
        calls.append("cpp")
        return service._not_connected("cpp missing")

    monkeypatch.setattr(service, "_suggest_with_llama_server", fake_server)
    monkeypatch.setattr(service, "_suggest_with_llama_cpp", fake_cpp)
    result = service.suggest_preset("непонятный запрос без шаблона", flat_preset())
    assert calls == ["server"]
    assert result.connected is False
    assert result.raw_json == "server missing"


def test_llama_server_args_prefer_single_cuda_slot(monkeypatch, tmp_path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    server.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    monkeypatch.setenv("AIEQ_LLAMA_SERVER_DEVICE", "CUDA0")
    monkeypatch.setenv("AIEQ_LLAMA_SERVER_PARALLEL", "1")
    monkeypatch.setenv("AIEQ_LLAMA_SERVER_FLASH_ATTN", "on")
    monkeypatch.setenv("AIEQ_LLAMA_SERVER_REASONING", "off")
    monkeypatch.setenv("AIEQ_LLAMA_SERVER_CACHE_RAM", "0")
    monkeypatch.setenv("AIEQ_LLAMA_N_GPU_LAYERS", "-1")

    args = AiEqualizerService()._llama_server_args(server, model)
    assert args[args.index("-ngl") + 1] == "all"
    assert args[args.index("-np") + 1] == "1"
    assert args[args.index("--device") + 1] == "CUDA0"
    assert args[args.index("-fa") + 1] == "on"
    assert args[args.index("-rea") + 1] == "off"
    assert "--reasoning-format" not in args
    assert "--reasoning-budget" not in args
    assert args[args.index("--cache-ram") + 1] == "0"
    assert "--no-webui" in args


def test_llama_server_args_enable_reasoning_by_default(tmp_path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    server.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")

    args = AiEqualizerService()._llama_server_args(server, model)
    assert args[args.index("-rea") + 1] == "auto"
    assert args[args.index("--reasoning-format") + 1] == "deepseek"
    assert args[args.index("--reasoning-budget") + 1] == "1024"
