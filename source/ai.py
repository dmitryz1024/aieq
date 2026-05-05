from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_env_file
from .models import Preset

AI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "assistant_message": {"type": "string"},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["peaking", "low_shelf", "high_shelf", "low_pass", "high_pass", "band_pass", "notch"],
                    },
                    "freq": {"type": "number"},
                    "q": {"type": "number"},
                    "gain": {"type": "number"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["type", "freq", "q", "gain", "enabled"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["name", "assistant_message", "filters"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are an expert parametric equalizer assistant for a Windows prototype.
Return only structured JSON matching the requested schema.
The application combines overlapping filters with envelope mixing, not cascade summing.
Therefore avoid stacking many similar filters on the same frequency range; choose decisive bands.
Use these filter types only: peaking, low_shelf, high_shelf, low_pass, high_pass, band_pass, notch.
Keep most musical changes within +/-6 dB unless the user asks for a radical effect.
Use Q around 0.6-1.0 for broad tone, 1.0-2.5 for focused tone, 4.0-10.0 for resonance/notch work.
Frequencies are in Hz. Preserve useful existing filters and produce a complete new preset.
You can use saved_presets as references when the user asks to modify or reuse an existing preset.
Preset name must be 1-3 concise meaningful words. Never use "New" and never start the name with "New".
Respond in Russian in assistant_message.
""".strip()

NOT_CONNECTED_MESSAGE = "Ваш ИИ-агент не подключен"
MODELS_DIR = Path("models")
MODEL_EXTENSIONS = (".gguf",)


def list_local_models(models_dir: Path = MODELS_DIR) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        (path for path in models_dir.iterdir() if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def sanitize_ai_preset_name(name: str, *, fallback: str = "AI Preset") -> str:
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)?", name.strip(), flags=re.UNICODE)
    clean = " ".join(words[:3]).strip()
    if not clean:
        clean = fallback
    if clean.casefold() == "new" or clean.casefold().startswith("new "):
        clean = fallback
    if clean.casefold() in {"preset", "ai", "eq", "equalizer"}:
        clean = fallback
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)?", clean.strip(), flags=re.UNICODE)
    return (" ".join(words[:3]).strip() or "AI Preset")[:80]


def fallback_ai_preset_name(user_text: str) -> str:
    text = user_text.casefold()
    if "v-" in text or "v " in text or "медиа" in text or "масс" in text:
        return "Media V Shape"
    if "бас" in text or "низ" in text:
        return "Bass Shape"
    if "вокал" in text or "голос" in text:
        return "Vocal Focus"
    if "ярк" in text or "верх" in text or "воздух" in text:
        return "Bright Air"
    if "мяг" in text or "резк" in text:
        return "Soft Tone"
    return "AI Preset"


@dataclass(slots=True)
class AiPresetResult:
    preset: Preset | None
    assistant_message: str
    used_model: str
    connected: bool = True
    raw_json: str | None = None


class AiEqualizerService:
    def __init__(self, model: str | None = None) -> None:
        self.model_override = model
        self.provider = "auto"
        self.openai_model = "gpt-5.2"
        self.llama_model_path = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
        self.llama_n_ctx = 4096
        self.llama_n_threads = max(1, (os.cpu_count() or 6) - 1)
        self.llama_n_gpu_layers = 0
        self.llama_max_tokens = 900
        self.llama_temperature = 0.2
        self._llama: Any | None = None
        self._llama_signature: tuple[str, int, int, int] | None = None
        self._refresh_config()

    def _refresh_config(self) -> None:
        load_env_file()
        self.provider = os.environ.get("AIEQ_AI_PROVIDER", "auto").strip().lower() or "auto"
        self.openai_model = self.model_override or os.environ.get("AIEQ_OPENAI_MODEL", "gpt-5.2")
        self.llama_model_path = Path(
            os.environ.get("AIEQ_LLAMA_MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
        ).expanduser()
        self.llama_n_ctx = self._env_int("AIEQ_LLAMA_N_CTX", 4096)
        self.llama_n_threads = self._env_int("AIEQ_LLAMA_N_THREADS", max(1, (os.cpu_count() or 6) - 1))
        self.llama_n_gpu_layers = self._env_int("AIEQ_LLAMA_N_GPU_LAYERS", 0)
        self.llama_max_tokens = self._env_int("AIEQ_LLAMA_MAX_TOKENS", 900)
        self.llama_temperature = self._env_float("AIEQ_LLAMA_TEMPERATURE", 0.2)

    def suggest_preset(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
    ) -> AiPresetResult:
        self._refresh_config()
        if self.provider == "none":
            return self._not_connected()
        if self.provider in {"llama_cpp", "llamacpp", "local"}:
            return self._suggest_with_llama_cpp(user_text, current_preset, saved_presets=saved_presets, model_path=model_path)
        if self.provider == "openai":
            return self._suggest_with_openai(user_text, current_preset, saved_presets=saved_presets)
        result = self._suggest_with_llama_cpp(user_text, current_preset, saved_presets=saved_presets, model_path=model_path)
        if result.connected:
            return result
        if os.environ.get("OPENAI_API_KEY"):
            result = self._suggest_with_openai(user_text, current_preset, saved_presets=saved_presets)
            if result.connected:
                return result
        return self._not_connected(result.raw_json)

    def _suggest_with_llama_cpp(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
    ) -> AiPresetResult:
        model_path = (model_path or self.llama_model_path).expanduser()
        if not model_path.exists():
            return self._not_connected(f"Model file not found: {model_path}")

        try:
            llm = self._load_llama_cpp_model(model_path)
        except Exception as exc:  # noqa: BLE001 - local model is optional.
            return self._not_connected(str(exc))

        payload = {
            "user_request": user_text,
            "current_preset": current_preset.to_dict(),
            "saved_presets": self._serialize_saved_presets(saved_presets),
            "language": "ru",
            "schema": AI_SCHEMA,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Return compact JSON only with keys: name, assistant_message, filters. "
                    "Every filter must contain type, freq, q, gain, enabled.\n\n"
                    f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            raw = self._llama_chat_json(llm, messages)
            data = self._loads_json_object(raw)
            name = sanitize_ai_preset_name(str(data["name"]), fallback=fallback_ai_preset_name(user_text))
            preset = Preset.from_dict({"name": name, "filters": data["filters"]}).sanitized()
            return AiPresetResult(
                preset=preset,
                assistant_message=str(data.get("assistant_message", "Готово, применяю новый пресет.")),
                used_model=f"llama-cpp:{model_path.name}",
                connected=True,
                raw_json=raw,
            )
        except Exception as exc:  # noqa: BLE001
            return self._not_connected(str(exc))

    def _load_llama_cpp_model(self, model_path: Path) -> Any:
        signature = (str(model_path.resolve()), self.llama_n_ctx, self.llama_n_threads, self.llama_n_gpu_layers)
        if self._llama is not None and self._llama_signature == signature:
            return self._llama

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed") from exc

        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=self.llama_n_ctx,
            n_threads=self.llama_n_threads,
            n_gpu_layers=self.llama_n_gpu_layers,
            verbose=False,
        )
        self._llama_signature = signature
        return self._llama

    def _llama_chat_json(self, llm: Any, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": self.llama_temperature,
            "max_tokens": self.llama_max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = llm.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("response_format", None)
            response = llm.create_chat_completion(**kwargs)
        content = response["choices"][0]["message"]["content"]
        return str(content).strip()

    def _suggest_with_openai(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
    ) -> AiPresetResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._not_connected()

        try:
            from openai import OpenAI
        except ImportError:
            return self._not_connected("openai package is not installed")

        payload = {
            "user_request": user_text,
            "current_preset": current_preset.to_dict(),
            "saved_presets": self._serialize_saved_presets(saved_presets),
            "language": "ru",
        }
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.openai_model,
                instructions=SYSTEM_PROMPT,
                input=json.dumps(payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "eq_preset_suggestion",
                        "strict": True,
                        "schema": AI_SCHEMA,
                    }
                },
            )
            raw = response.output_text
            data = json.loads(raw)
            name = sanitize_ai_preset_name(str(data["name"]), fallback=fallback_ai_preset_name(user_text))
            preset = Preset.from_dict({"name": name, "filters": data["filters"]}).sanitized()
            return AiPresetResult(
                preset=preset,
                assistant_message=str(data["assistant_message"]),
                used_model=self.openai_model,
                connected=True,
                raw_json=raw,
            )
        except Exception as exc:  # noqa: BLE001 - UI should keep working offline.
            return self._not_connected(str(exc))

    def _loads_json_object(self, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("AI response is not a JSON object")
        return data

    def _serialize_saved_presets(self, presets: list[Preset] | None) -> list[dict[str, Any]]:
        if not presets:
            return []
        return [preset.to_dict(include_id=True) for preset in presets[:12]]

    def _not_connected(self, raw_json: str | None = None) -> AiPresetResult:
        return AiPresetResult(
            preset=None,
            assistant_message=NOT_CONNECTED_MESSAGE,
            used_model="none",
            connected=False,
            raw_json=raw_json,
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            return default
