from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
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
Return only structured data matching the schema.
The application combines overlapping filters with envelope mixing, not cascade summing.
Therefore avoid stacking many similar filters on the same frequency range; choose decisive bands.
Use these filter types only: peaking, low_shelf, high_shelf, low_pass, high_pass, band_pass, notch.
Keep most musical changes within +/-6 dB unless the user asks for a radical effect.
Use Q around 0.6-1.0 for broad tone, 1.0-2.5 for focused tone, 4.0-10.0 for resonance/notch work.
Frequencies are in Hz. Preserve useful existing filters and produce a complete new preset.
""".strip()


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
        self.model = "gpt-5.2"
        self.ollama_base_url = "http://127.0.0.1:11434"
        self.ollama_model = ""
        self.ollama_timeout = 300.0
        self._refresh_config()

    def _refresh_config(self) -> None:
        load_env_file()
        self.provider = os.environ.get("AIEQ_AI_PROVIDER", "auto").strip().lower() or "auto"
        self.model = self.model_override or os.environ.get("AIEQ_OPENAI_MODEL", "gpt-5.2")
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()
        self.ollama_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "300"))

    def suggest_preset(self, user_text: str, current_preset: Preset) -> AiPresetResult:
        self._refresh_config()
        if self.provider == "none":
            return self._not_connected()
        if self.provider == "ollama":
            return self._suggest_with_ollama(user_text, current_preset)
        if self.provider == "openai":
            return self._suggest_with_openai(user_text, current_preset)

        if os.environ.get("OPENAI_API_KEY"):
            result = self._suggest_with_openai(user_text, current_preset)
            if result.connected:
                return result
        return self._suggest_with_ollama(user_text, current_preset)

    def _suggest_with_openai(self, user_text: str, current_preset: Preset) -> AiPresetResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._not_connected()

        try:
            from openai import OpenAI
        except ImportError:
            return self._not_connected()

        payload = {
            "user_request": user_text,
            "current_preset": current_preset.to_dict(),
            "language": "ru",
        }
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.model,
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
            preset = Preset.from_dict(
                {
                    "name": data["name"],
                    "filters": data["filters"],
                }
            ).sanitized()
            return AiPresetResult(
                preset=preset,
                assistant_message=str(data["assistant_message"]),
                used_model=self.model,
                connected=True,
                raw_json=raw,
            )
        except Exception as exc:  # noqa: BLE001 - UI should keep working offline.
            return self._not_connected(raw_json=str(exc))

    def _suggest_with_ollama(self, user_text: str, current_preset: Preset) -> AiPresetResult:
        model = self.ollama_model or self._first_ollama_model()
        if not model:
            return self._not_connected()

        payload = {
            "user_request": user_text,
            "current_preset": current_preset.to_dict(),
            "language": "ru",
        }
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Return compact JSON only with keys: name, assistant_message, filters. "
            "Every filter must contain type, freq, q, gain, enabled.\n\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = self._ollama_post(
                "/api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2},
                },
            )
            raw = str(response.get("response", "")).strip()
            data = self._loads_json_object(raw)
            preset = Preset.from_dict({"name": data["name"], "filters": data["filters"]}).sanitized()
            return AiPresetResult(
                preset=preset,
                assistant_message=str(data.get("assistant_message", "Готово, применяю новый пресет.")),
                used_model=f"ollama:{model}",
                connected=True,
                raw_json=raw,
            )
        except Exception as exc:  # noqa: BLE001 - local model may be stopped or missing.
            return self._not_connected(raw_json=str(exc))

    def _first_ollama_model(self) -> str | None:
        try:
            data = self._ollama_get("/api/tags")
            models = data.get("models", [])
            if not isinstance(models, list) or not models:
                return None
            name = models[0].get("name")
            return str(name) if name else None
        except Exception:  # noqa: BLE001
            return None

    def _ollama_get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.ollama_base_url}{path}", timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _ollama_post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.ollama_base_url}{path}",
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.ollama_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable: {exc}") from exc

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

    def _not_connected(self, raw_json: str | None = None) -> AiPresetResult:
        return AiPresetResult(
            preset=None,
            assistant_message="Ваш ИИ-агент не подключен",
            used_model="none",
            connected=False,
            raw_json=raw_json,
        )
