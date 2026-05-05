from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_env_file
from .curves import FrequencyCurve
from .models import Preset

AI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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
    "required": ["assistant_message", "filters"],
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
If the user gives explicit frequency, gain, or Q values, follow those values unless they are unsafe or outside limits.
Frequencies are in Hz. Preserve useful existing filters and produce a complete new preset.
You can use saved_presets as references when the user asks to modify or reuse an existing preset.
Use current_device as the measured response of the selected playback device. If current_device.raw_txt is available,
treat it as the source-of-truth frequency response data and compensate with respect to that curve.
Respond in Russian in assistant_message. Make assistant_message a brief 1-2 sentence summary of the decisions,
without listing filter parameters or describing every filter.
""".strip()

NOT_CONNECTED_MESSAGE = "Ваш ИИ-агент не подключен"
MODELS_DIR = Path("models")
MODEL_EXTENSIONS = (".gguf",)
DEFAULT_LLAMA_N_CTX = 8192
MAX_SAVED_PRESETS = 8
MAX_SAVED_PRESET_FILTERS = 14
MAX_DEVICE_RAW_LINES = 48
MAX_DEVICE_RAW_CHARS = 2500
MAX_DEVICE_POINTS = 48
COMPACT_SAVED_PRESETS = 4
COMPACT_SAVED_PRESET_FILTERS = 10
COMPACT_DEVICE_RAW_LINES = 24
COMPACT_DEVICE_RAW_CHARS = 1200
COMPACT_DEVICE_POINTS = 24


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
        self.llama_n_ctx = DEFAULT_LLAMA_N_CTX
        self.llama_n_threads = max(1, (os.cpu_count() or 6) - 1)
        self.llama_n_gpu_layers = -1
        self.llama_n_batch = 512
        self.llama_max_tokens = 900
        self.llama_temperature = 0.2
        self._llama: Any | None = None
        self._llama_signature: tuple[str, int, int, int, int] | None = None
        self._refresh_config()

    def _refresh_config(self) -> None:
        load_env_file()
        self.provider = os.environ.get("AIEQ_AI_PROVIDER", "auto").strip().lower() or "auto"
        self.openai_model = self.model_override or os.environ.get("AIEQ_OPENAI_MODEL", "gpt-5.2")
        self.llama_model_path = Path(
            os.environ.get("AIEQ_LLAMA_MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
        ).expanduser()
        self.llama_n_ctx = self._env_int("AIEQ_LLAMA_N_CTX", DEFAULT_LLAMA_N_CTX)
        self.llama_n_threads = self._env_int("AIEQ_LLAMA_N_THREADS", max(1, (os.cpu_count() or 6) - 1))
        self.llama_n_gpu_layers = self._env_int("AIEQ_LLAMA_N_GPU_LAYERS", -1)
        self.llama_n_batch = self._env_int("AIEQ_LLAMA_N_BATCH", 512)
        self.llama_max_tokens = self._env_int("AIEQ_LLAMA_MAX_TOKENS", 900)
        self.llama_temperature = self._env_float("AIEQ_LLAMA_TEMPERATURE", 0.2)

    def suggest_preset(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
        device_curve: FrequencyCurve | None = None,
    ) -> AiPresetResult:
        self._refresh_config()
        if self.provider == "none":
            return self._not_connected()
        if self.provider in {"llama_cpp", "llamacpp", "local"}:
            return self._suggest_with_llama_cpp(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                model_path=model_path,
                device_curve=device_curve,
            )
        if self.provider == "openai":
            return self._suggest_with_openai(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
            )
        result = self._suggest_with_llama_cpp(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            model_path=model_path,
            device_curve=device_curve,
        )
        if result.connected:
            return result
        if os.environ.get("OPENAI_API_KEY"):
            result = self._suggest_with_openai(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
            )
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
        device_curve: FrequencyCurve | None = None,
    ) -> AiPresetResult:
        model_path = (model_path or self.llama_model_path).expanduser()
        if not model_path.exists():
            return self._not_connected(f"Model file not found: {model_path}")

        try:
            llm = self._load_llama_cpp_model(model_path)
        except Exception as exc:  # noqa: BLE001 - local model is optional.
            return self._not_connected(str(exc))

        try:
            return self._run_llama_request(
                llm,
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
                model_path=model_path,
                compact=False,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_context_limit_error(exc):
                return self._not_connected(str(exc))
            self.clear_context()
            try:
                llm = self._load_llama_cpp_model(model_path)
                return self._run_llama_request(
                    llm,
                    user_text,
                    current_preset,
                    saved_presets=saved_presets,
                    device_curve=device_curve,
                    model_path=model_path,
                    compact=True,
                )
            except Exception as retry_exc:  # noqa: BLE001
                return self._not_connected(str(retry_exc))

    def _run_llama_request(
        self,
        llm: Any,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None,
        device_curve: FrequencyCurve | None,
        model_path: Path,
        compact: bool,
    ) -> AiPresetResult:
        payload = self._build_llama_payload(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            device_curve=device_curve,
            compact=compact,
        )
        raw = self._llama_chat_json(llm, self._llama_messages(payload))
        data = self._loads_json_object(raw)
        preset = Preset.from_dict({"name": "AI Preset", "filters": data["filters"]}).sanitized()
        return AiPresetResult(
            preset=preset,
            assistant_message=str(data.get("assistant_message", "Готово, применяю новый пресет.")),
            used_model=f"llama-cpp:{model_path.name}",
            connected=True,
            raw_json=raw,
        )

    def _build_llama_payload(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None,
        device_curve: FrequencyCurve | None,
        compact: bool,
    ) -> dict[str, Any]:
        return {
            "user_request": user_text,
            "current_preset": self._serialize_preset(
                current_preset,
                include_id=True,
                max_filters=COMPACT_SAVED_PRESET_FILTERS if compact else MAX_SAVED_PRESET_FILTERS,
            ),
            "saved_presets": self._serialize_saved_presets(saved_presets, user_text=user_text, compact=compact),
            "current_device": self._serialize_device_curve(device_curve, compact=compact),
            "language": "ru",
            "output_contract": "JSON object with assistant_message and filters; every filter has type, freq, q, gain, enabled.",
            "compact_context": compact,
        }

    @staticmethod
    def _llama_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Return compact JSON only with keys: assistant_message, filters. "
                    "Every filter must contain type, freq, q, gain, enabled. "
                    "assistant_message must summarize the tonal intent only, with no per-filter values.\n\n"
                    f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]

    def _load_llama_cpp_model(self, model_path: Path) -> Any:
        gpu_layers = self._effective_gpu_layers(self.llama_n_gpu_layers)
        signature = (
            str(model_path.resolve()),
            self.llama_n_ctx,
            self.llama_n_threads,
            gpu_layers,
            self.llama_n_batch,
        )
        if self._llama is not None and self._llama_signature == signature:
            return self._llama

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed") from exc

        try:
            return self._create_llama_cpp_model(Llama, model_path, gpu_layers, signature)
        except Exception:
            if gpu_layers == 0:
                raise

        cpu_signature = (
            str(model_path.resolve()),
            self.llama_n_ctx,
            self.llama_n_threads,
            0,
            self.llama_n_batch,
        )
        if self._llama is not None and self._llama_signature == cpu_signature:
            return self._llama
        return self._create_llama_cpp_model(Llama, model_path, 0, cpu_signature)

    def _create_llama_cpp_model(
        self,
        llama_class: Any,
        model_path: Path,
        gpu_layers: int,
        signature: tuple[str, int, int, int, int],
    ) -> Any:
        self._llama = llama_class(
            model_path=str(model_path),
            n_ctx=self.llama_n_ctx,
            n_threads=self.llama_n_threads,
            n_gpu_layers=gpu_layers,
            n_batch=self.llama_n_batch,
            verbose=False,
        )
        self._llama_signature = signature
        return self._llama

    def _effective_gpu_layers(self, requested_gpu_layers: int) -> int:
        if requested_gpu_layers == 0:
            return 0
        return requested_gpu_layers if self._llama_supports_gpu_offload() else 0

    @staticmethod
    def _llama_supports_gpu_offload() -> bool:
        try:
            from llama_cpp import llama_cpp as llama_cpp_lib
        except Exception:  # noqa: BLE001 - the import error is handled by model loading.
            return True
        supports = getattr(llama_cpp_lib, "llama_supports_gpu_offload", None)
        if not callable(supports):
            return True
        try:
            return bool(supports())
        except Exception:  # noqa: BLE001 - if detection fails, try loading and let it report.
            return True

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
        device_curve: FrequencyCurve | None = None,
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
            "current_device": self._serialize_device_curve(device_curve),
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
            preset = Preset.from_dict({"name": "AI Preset", "filters": data["filters"]}).sanitized()
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

    def clear_context(self) -> None:
        llama = self._llama
        self._llama = None
        self._llama_signature = None
        if llama is None:
            return
        reset = getattr(llama, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001 - context clearing is best effort.
                pass

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return (
            "exceed context" in message
            or "context window" in message
            or "requested tokens" in message
            or "n_ctx" in message
        )

    def _serialize_saved_presets(
        self,
        presets: list[Preset] | None,
        *,
        user_text: str = "",
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        if not presets:
            return []
        limit = COMPACT_SAVED_PRESETS if compact else MAX_SAVED_PRESETS
        max_filters = COMPACT_SAVED_PRESET_FILTERS if compact else MAX_SAVED_PRESET_FILTERS
        serialized: list[dict[str, Any]] = []
        for preset in self._prioritize_saved_presets(presets, user_text)[:limit]:
            serialized.append(self._serialize_preset(preset, include_id=True, max_filters=max_filters))
        return serialized

    @staticmethod
    def _serialize_preset(preset: Preset, *, include_id: bool, max_filters: int) -> dict[str, Any]:
        data = preset.to_dict(include_id=include_id)
        data["filters"] = data["filters"][:max_filters]
        data["filters_truncated"] = len(preset.filters) > max_filters
        return data

    @staticmethod
    def _prioritize_saved_presets(presets: list[Preset], user_text: str) -> list[Preset]:
        if not user_text:
            return presets
        lowered_text = user_text.casefold()

        def rank(item: tuple[int, Preset]) -> tuple[int, int]:
            index, preset = item
            name = preset.name.casefold()
            if name and name in lowered_text:
                return (0, index)
            name_tokens = [token for token in re.findall(r"[^\W_]+", name, flags=re.UNICODE) if len(token) >= 4]
            if name_tokens and any(token in lowered_text for token in name_tokens):
                return (1, index)
            return (2, index)

        return [preset for _index, preset in sorted(enumerate(presets), key=rank)]

    def _serialize_device_curve(self, curve: FrequencyCurve | None, *, compact: bool = False) -> dict[str, Any]:
        if curve is None:
            return {"name": "Default", "raw_txt": "", "points": []}
        raw_txt = ""
        raw_txt_is_sampled = False
        if curve.path is not None and curve.path.exists():
            try:
                raw_txt, raw_txt_is_sampled = self._sample_raw_curve_text(
                    curve.path.read_text(encoding="utf-8-sig"),
                    compact=compact,
                )
            except OSError:
                raw_txt = ""
        return {
            "name": curve.name,
            "path": str(curve.path) if curve.path is not None else "",
            "raw_txt": raw_txt,
            "raw_txt_is_sampled": raw_txt_is_sampled,
            "points": self._sample_curve_points(curve, compact=compact),
        }

    @staticmethod
    def _sample_raw_curve_text(text: str, *, compact: bool = False) -> tuple[str, bool]:
        max_lines = COMPACT_DEVICE_RAW_LINES if compact else MAX_DEVICE_RAW_LINES
        max_chars = COMPACT_DEVICE_RAW_CHARS if compact else MAX_DEVICE_RAW_CHARS
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sampled = False
        if len(lines) > max_lines:
            lines = AiEqualizerService._even_sample(lines, max_lines)
            sampled = True
        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars].rsplit("\n", 1)[0]
            sampled = True
        return result, sampled

    @staticmethod
    def _sample_curve_points(curve: FrequencyCurve, *, compact: bool = False) -> list[dict[str, float]]:
        max_points = COMPACT_DEVICE_POINTS if compact else MAX_DEVICE_POINTS
        freqs = curve.freqs.tolist()
        values = curve.db.tolist()
        points = list(zip(freqs, values, strict=False))
        if len(points) > max_points:
            points = AiEqualizerService._even_sample(points, max_points)
        return [
            {"freq": round(float(freq), 3), "db": round(float(db), 3)}
            for freq, db in points
        ]

    @staticmethod
    def _even_sample(items: list[Any], limit: int) -> list[Any]:
        if len(items) <= limit:
            return items
        if limit <= 1:
            return items[:limit]
        last_index = len(items) - 1
        indexes = [round(index * last_index / (limit - 1)) for index in range(limit)]
        return [items[index] for index in indexes]

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
