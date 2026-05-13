from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
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
Think carefully before answering, but return only structured JSON matching the requested schema.
The application combines overlapping filters with envelope mixing, not cascade summing.
Therefore avoid stacking many similar filters on the same frequency range; choose decisive bands.
Use these filter types only: peaking, low_shelf, high_shelf, low_pass, high_pass, band_pass, notch.
Keep most musical changes within +/-6 dB unless the user asks for a radical effect.
Use Q around 0.6-1.0 for broad tone, 1.0-2.5 for focused tone, 4.0-10.0 for resonance/notch work.
If the user gives explicit frequency, gain, or Q values, follow those values unless they are unsafe or outside limits.
Frequencies are in Hz. Preserve useful existing filters and produce a complete new preset.
For exact requests, include filters that match the requested frequencies, gains, and Q values as closely as possible.
For vague requests, infer a tasteful complete tonal profile: 3-7 enabled filters is usually enough.
Use broad shelves and broad peaking filters for musical tone shaping; use narrow filters only for resonances,
harshness, boxiness, sibilance, rumble, or hum. Avoid low_pass/high_pass unless the user asks for a cutoff effect.
If the user asks for a common style, translate it into a concrete EQ curve:
soft/smooth = less 2.5-6 kHz harshness and controlled air; warm = low-mid body without muddy bass;
V-shaped/media = bass and air lift with controlled mids; clear/vocal = less mud and more presence without sharpness.
You can use referenced_preset only when the user explicitly asks to modify or reuse that preset.
If referenced_preset is null, do not copy any saved preset. Create a fresh preset from the user request,
the current preset, current_device, and conversation_history. If referenced_preset is present, treat it as
the explicit starting point, but still make the requested changes instead of returning it unchanged.
Never return current_preset or referenced_preset unchanged as your answer.
Use current_device as the measured response of the selected playback device. If current_device.raw_txt is available,
treat it as the source-of-truth frequency response data and compensate with respect to that curve.
First compensate obvious current_device problems, then apply the user's taste request on top of that.
Respond in Russian in assistant_message. Make assistant_message a brief 1-2 sentence summary of the decisions,
without listing filter parameters or describing every filter.
""".strip()

NOT_CONNECTED_MESSAGE = "Ваш ИИ-агент не подключен"
MODELS_DIR = Path("models")
MODEL_EXTENSIONS = (".gguf",)
DEFAULT_LLAMA_N_CTX = 12288
DEFAULT_LLAMA_N_THREADS = min(8, max(1, os.cpu_count() or 8))
DEFAULT_LLAMA_MAX_TOKENS = 2048
DEFAULT_LLAMA_TEMPERATURE = 0.35
MAX_SAVED_PRESET_FILTERS = 14
MAX_DEVICE_POINTS = 48
COMPACT_SAVED_PRESET_FILTERS = 10
COMPACT_DEVICE_RAW_LINES = 24
COMPACT_DEVICE_RAW_CHARS = 1200
COMPACT_DEVICE_POINTS = 24
COMPACT_CHAT_MESSAGES = 10


def list_local_models(models_dir: Path = MODELS_DIR) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        (path for path in models_dir.iterdir() if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


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
        self.ai_allow_cpu_fallback = True
        self.openai_model = "gpt-5.2"
        self.llama_model_path = Path("models/Qwen3-4B-Q4_K_M.gguf")
        self.llama_n_ctx = DEFAULT_LLAMA_N_CTX
        self.llama_n_threads = DEFAULT_LLAMA_N_THREADS
        self.llama_n_gpu_layers = -1
        self.llama_n_batch = 1024
        self.llama_max_tokens = DEFAULT_LLAMA_MAX_TOKENS
        self.llama_temperature = DEFAULT_LLAMA_TEMPERATURE
        self.llama_server_path = Path("runtime/llama.cpp/llama-server.exe")
        self.llama_server_host = "127.0.0.1"
        self.llama_server_port = 8080
        self.llama_server_auto_start = True
        self.llama_server_startup_timeout = 45.0
        self.llama_server_device = "auto"
        self.llama_server_parallel = 1
        self.llama_server_flash_attn = "on"
        self.llama_server_reasoning = "auto"
        self.llama_server_reasoning_format = "deepseek"
        self.llama_server_reasoning_budget = 1024
        self.llama_server_cache_ram = 0
        self.llama_server_require_gpu = False
        self.llama_server_extra_args = ""
        self.llama_server_log_path = self._default_llama_server_log_path()
        self._llama_server_process: subprocess.Popen[Any] | None = None
        self._llama_server_log_file: Any | None = None
        self._llama: Any | None = None
        self._llama_signature: tuple[str, int, int, int, int] | None = None
        self._refresh_config()

    def _refresh_config(self) -> None:
        load_env_file()
        self.provider = os.environ.get("AIEQ_AI_PROVIDER", "auto").strip().lower() or "auto"
        self.ai_allow_cpu_fallback = self._env_bool("AIEQ_AI_ALLOW_CPU_FALLBACK", True)
        self.openai_model = self.model_override or os.environ.get("AIEQ_OPENAI_MODEL", "gpt-5.2")
        self.llama_model_path = Path(
            os.environ.get("AIEQ_LLAMA_MODEL_PATH", "models/Qwen3-4B-Q4_K_M.gguf")
        ).expanduser()
        self.llama_n_ctx = self._env_int("AIEQ_LLAMA_N_CTX", DEFAULT_LLAMA_N_CTX)
        self.llama_n_threads = self._env_int("AIEQ_LLAMA_N_THREADS", DEFAULT_LLAMA_N_THREADS)
        self.llama_n_gpu_layers = self._env_int("AIEQ_LLAMA_N_GPU_LAYERS", -1)
        self.llama_n_batch = self._env_int("AIEQ_LLAMA_N_BATCH", 1024)
        self.llama_max_tokens = self._env_int("AIEQ_LLAMA_MAX_TOKENS", DEFAULT_LLAMA_MAX_TOKENS)
        self.llama_temperature = self._env_float("AIEQ_LLAMA_TEMPERATURE", DEFAULT_LLAMA_TEMPERATURE)
        self.llama_server_path = Path(
            os.environ.get("AIEQ_LLAMA_SERVER_PATH", "runtime/llama.cpp/llama-server.exe")
        ).expanduser()
        self.llama_server_host = os.environ.get("AIEQ_LLAMA_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.llama_server_port = self._env_int("AIEQ_LLAMA_SERVER_PORT", 8080)
        self.llama_server_auto_start = self._env_bool("AIEQ_LLAMA_SERVER_AUTO_START", True)
        self.llama_server_startup_timeout = self._env_float("AIEQ_LLAMA_SERVER_STARTUP_TIMEOUT", 45.0)
        self.llama_server_device = os.environ.get("AIEQ_LLAMA_SERVER_DEVICE", "auto").strip() or "auto"
        self.llama_server_parallel = max(1, self._env_int("AIEQ_LLAMA_SERVER_PARALLEL", 1))
        self.llama_server_flash_attn = os.environ.get("AIEQ_LLAMA_SERVER_FLASH_ATTN", "on").strip()
        self.llama_server_reasoning = os.environ.get("AIEQ_LLAMA_SERVER_REASONING", "auto").strip()
        self.llama_server_reasoning_format = os.environ.get("AIEQ_LLAMA_SERVER_REASONING_FORMAT", "deepseek").strip()
        self.llama_server_reasoning_budget = self._env_int("AIEQ_LLAMA_SERVER_REASONING_BUDGET", 1024)
        self.llama_server_cache_ram = self._env_int("AIEQ_LLAMA_SERVER_CACHE_RAM", 0)
        self.llama_server_require_gpu = self._env_bool("AIEQ_LLAMA_SERVER_REQUIRE_GPU", False)
        self.llama_server_extra_args = os.environ.get("AIEQ_LLAMA_SERVER_EXTRA_ARGS", "").strip()
        log_path = os.environ.get("AIEQ_LLAMA_SERVER_LOG_PATH", "").strip()
        self.llama_server_log_path = Path(log_path).expanduser() if log_path else self._default_llama_server_log_path()

    def suggest_preset(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
        device_curve: FrequencyCurve | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AiPresetResult:
        self._refresh_config()
        if self.provider == "none":
            return self._not_connected()
        if self.provider in {"llama_server", "llama-server", "server"}:
            return self._suggest_with_llama_server(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                model_path=model_path,
                device_curve=device_curve,
                chat_history=chat_history,
            )
        if self.provider in {"llama_cpp", "llamacpp", "local"}:
            return self._suggest_with_llama_cpp(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                model_path=model_path,
                device_curve=device_curve,
                chat_history=chat_history,
            )
        if self.provider == "openai":
            return self._suggest_with_openai(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
                chat_history=chat_history,
            )
        result = self._suggest_with_llama_server(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            model_path=model_path,
            device_curve=device_curve,
            chat_history=chat_history,
        )
        if result.connected:
            return result
        if not self.ai_allow_cpu_fallback:
            if os.environ.get("OPENAI_API_KEY"):
                result = self._suggest_with_openai(
                    user_text,
                    current_preset,
                    saved_presets=saved_presets,
                    device_curve=device_curve,
                    chat_history=chat_history,
                )
                if result.connected:
                    return result
            return self._not_connected(result.raw_json)
        result = self._suggest_with_llama_cpp(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            model_path=model_path,
            device_curve=device_curve,
            chat_history=chat_history,
        )
        if result.connected:
            return result
        if os.environ.get("OPENAI_API_KEY"):
            result = self._suggest_with_openai(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
                chat_history=chat_history,
            )
            if result.connected:
                return result
        return self._not_connected(result.raw_json)

    def _suggest_with_llama_server(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
        device_curve: FrequencyCurve | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AiPresetResult:
        model_path = (model_path or self.llama_model_path).expanduser()
        if not model_path.exists():
            return self._not_connected(f"Model file not found: {model_path}")
        try:
            self._ensure_llama_server(model_path)
            return self._run_llama_server_request(
                user_text,
                current_preset,
                saved_presets=saved_presets,
                device_curve=device_curve,
                model_path=model_path,
                compact=False,
                chat_history=chat_history,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._is_context_limit_error(exc):
                return self._not_connected(str(exc))
            try:
                return self._run_llama_server_request(
                    user_text,
                    current_preset,
                    saved_presets=saved_presets,
                    device_curve=device_curve,
                    model_path=model_path,
                    compact=True,
                    chat_history=chat_history,
                )
            except Exception as retry_exc:  # noqa: BLE001
                return self._not_connected(str(retry_exc))

    def _run_llama_server_request(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None,
        device_curve: FrequencyCurve | None,
        model_path: Path,
        compact: bool,
        chat_history: list[dict[str, str]] | None,
    ) -> AiPresetResult:
        payload = self._build_llama_payload(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            device_curve=device_curve,
            compact=compact,
            chat_history=chat_history,
        )
        raw = self._llama_server_chat_json(self._llama_messages(payload), model_path)
        data = self._loads_json_object(raw)
        preset = Preset.from_dict({"name": "AI Preset", "filters": data["filters"]}).sanitized()
        return AiPresetResult(
            preset=preset,
            assistant_message=str(data.get("assistant_message", "Готово, применяю новый пресет.")),
            used_model=f"llama-server:{model_path.name}",
            connected=True,
            raw_json=raw,
        )

    def _ensure_llama_server(self, model_path: Path) -> None:
        if self._llama_server_is_alive():
            if self.llama_server_require_gpu:
                server_path = self._resolve_llama_server_path()
                if server_path is None:
                    raise RuntimeError("llama-server GPU check failed: runtime executable was not found")
                self._ensure_requested_server_device(server_path)
                self._ensure_running_server_matches_config(model_path)
            return
        if not self.llama_server_auto_start:
            raise RuntimeError("llama-server is not running")
        server_path = self._resolve_llama_server_path()
        if server_path is None:
            raise RuntimeError("llama-server.exe not found; install llama.cpp runtime or use llama_cpp provider")

        if self.llama_server_require_gpu:
            self._ensure_requested_server_device(server_path)

        args = self._llama_server_args(server_path, model_path)
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        log_file = self._open_llama_server_log()
        self._llama_server_process = subprocess.Popen(  # noqa: S603 - user-configured local executable.
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(server_path.parent),
            env=self._llama_server_env(server_path),
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        deadline = time.monotonic() + self.llama_server_startup_timeout
        while time.monotonic() < deadline:
            if self._llama_server_process.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            if self._llama_server_is_alive():
                return
            time.sleep(0.35)
        raise RuntimeError("llama-server startup timed out")

    def _open_llama_server_log(self) -> Any:
        if self._llama_server_log_file is not None and not self._llama_server_log_file.closed:
            return self._llama_server_log_file
        self.llama_server_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._llama_server_log_file = self.llama_server_log_path.open("a", encoding="utf-8")
        self._llama_server_log_file.write(f"\n\n=== AIEQ llama-server {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self._llama_server_log_file.flush()
        return self._llama_server_log_file

    def _llama_server_args(self, server_path: Path, model_path: Path) -> list[str]:
        args = [
            str(server_path.resolve()),
            "-m",
            str(model_path.resolve()),
            "-c",
            str(self.llama_n_ctx),
            "-b",
            str(self.llama_n_batch),
            "-ub",
            str(min(512, max(1, self.llama_n_batch))),
            "-t",
            str(self.llama_n_threads),
            "-tb",
            str(self.llama_n_threads),
            "-ngl",
            self._server_gpu_layers(),
            "-np",
            str(self.llama_server_parallel),
            "-a",
            model_path.name,
            "--host",
            self.llama_server_host,
            "--port",
            str(self.llama_server_port),
            "--threads-http",
            "3",
            "--timeout",
            "600",
            "--no-webui",
            "--log-timestamps",
            "--log-prefix",
        ]
        if self.llama_server_device and self.llama_server_device.casefold() != "auto":
            args.extend(["--device", self.llama_server_device])
        if self.llama_server_flash_attn:
            args.extend(["-fa", self.llama_server_flash_attn])
        reasoning_mode = self.llama_server_reasoning.casefold()
        if self.llama_server_reasoning:
            args.extend(["-rea", self.llama_server_reasoning])
        if reasoning_mode != "off" and self.llama_server_reasoning_format:
            args.extend(["--reasoning-format", self.llama_server_reasoning_format])
        if reasoning_mode != "off" and self.llama_server_reasoning_budget >= 0:
            args.extend(["--reasoning-budget", str(self.llama_server_reasoning_budget)])
        if self.llama_server_cache_ram >= 0:
            args.extend(["--cache-ram", str(self.llama_server_cache_ram)])
        if self.llama_server_extra_args:
            args.extend(shlex.split(self.llama_server_extra_args, posix=os.name != "nt"))
        return args

    def _ensure_requested_server_device(self, server_path: Path) -> None:
        report = self._llama_server_device_report(server_path)
        if self._server_device_report_matches(report):
            return
        requested = self.llama_server_device if self.llama_server_device.casefold() != "auto" else "CUDA/Vulkan GPU"
        raise RuntimeError(
            f"llama-server does not see requested device {requested}. "
            "Install the llama.cpp CUDA DLL archive next to llama-server.exe and rerun "
            "`runtime\\llama.cpp\\llama-server.exe --list-devices`."
        )

    def _ensure_running_server_matches_config(self, model_path: Path) -> None:
        props = self._http_json("GET", self._llama_server_url("/props"), timeout=2.0)
        slots = props.get("total_slots")
        if isinstance(slots, int) and slots != self.llama_server_parallel:
            raise RuntimeError(
                f"running llama-server has {slots} slots; stop it and restart with -np {self.llama_server_parallel}"
            )
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict):
            n_ctx = settings.get("n_ctx")
            if isinstance(n_ctx, int) and n_ctx != self.llama_n_ctx:
                raise RuntimeError(
                    f"running llama-server uses n_ctx={n_ctx}; stop it and restart with -c {self.llama_n_ctx}"
                )
        alias = str(props.get("model_alias", ""))
        loaded_path = str(props.get("model_path", ""))
        if alias and alias != model_path.name and model_path.name not in loaded_path:
            raise RuntimeError(
                f"running llama-server loaded {alias}; stop it and restart with {model_path.name}"
            )

    def _llama_server_device_report(self, server_path: Path) -> str:
        try:
            completed = subprocess.run(  # noqa: S603 - local configured executable.
                [str(server_path.resolve()), "--list-devices"],
                cwd=str(server_path.parent),
                env=self._llama_server_env(server_path),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"llama-server GPU check failed: {exc}") from exc
        return f"{completed.stdout}\n{completed.stderr}"

    def _server_device_report_matches(self, report: str) -> bool:
        folded = report.casefold()
        requested = self.llama_server_device.casefold()
        if requested and requested != "auto":
            return requested in folded
        return "cuda" in folded or "vulkan" in folded

    @staticmethod
    def _llama_server_env(server_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        runtime_dir = str(server_path.parent)
        env["PATH"] = runtime_dir + os.pathsep + env.get("PATH", "")
        return env

    @staticmethod
    def _default_llama_server_log_path() -> Path:
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "AIEQ" / "logs" / "llama-server.log"

    def _resolve_llama_server_path(self) -> Path | None:
        candidates = [
            self.llama_server_path,
            Path("llama-server.exe"),
            Path("runtime/llama.cpp/llama-server.exe"),
            Path("runtime/llama.cpp/cuda/llama-server.exe"),
            Path("runtime/llama.cpp/vulkan/llama-server.exe"),
            Path("runtime/llama.cpp/cpu/llama-server.exe"),
        ]
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded.exists():
                return expanded.resolve()
        found = shutil.which("llama-server.exe") or shutil.which("llama-server")
        if found:
            return Path(found).resolve()
        return None

    def _llama_server_is_alive(self) -> bool:
        try:
            self._http_json("GET", self._llama_server_url("/health"), timeout=1.5)
            return True
        except Exception:  # noqa: BLE001
            try:
                self._http_json("GET", self._llama_server_url("/v1/models"), timeout=1.5)
                return True
            except Exception:  # noqa: BLE001
                return False

    def _llama_server_chat_json(self, messages: list[dict[str, str]], model_path: Path) -> str:
        request = {
            "model": model_path.name,
            "messages": messages,
            "temperature": self.llama_temperature,
            "max_tokens": self.llama_max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._http_json("POST", self._llama_server_url("/v1/chat/completions"), data=request, timeout=240.0)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama-server returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("llama-server returned an invalid message")
        content = message.get("content", "")
        return str(content).strip()

    def _llama_server_url(self, path: str) -> str:
        return f"http://{self.llama_server_host}:{self.llama_server_port}{path}"

    def _http_json(self, method: str, url: str, *, data: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local configurable endpoint.
                content = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
        return json.loads(content or "{}")

    def _server_gpu_layers(self) -> str:
        return "all" if self.llama_n_gpu_layers < 0 else str(self.llama_n_gpu_layers)

    def _suggest_with_llama_cpp(
        self,
        user_text: str,
        current_preset: Preset,
        *,
        saved_presets: list[Preset] | None = None,
        model_path: Path | None = None,
        device_curve: FrequencyCurve | None = None,
        chat_history: list[dict[str, str]] | None = None,
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
                chat_history=chat_history,
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
                    chat_history=chat_history,
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
        chat_history: list[dict[str, str]] | None,
    ) -> AiPresetResult:
        payload = self._build_llama_payload(
            user_text,
            current_preset,
            saved_presets=saved_presets,
            device_curve=device_curve,
            compact=compact,
            chat_history=chat_history,
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
        chat_history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        referenced_preset = self._serialize_referenced_preset(saved_presets, user_text)
        return {
            "user_request": user_text,
            "current_preset": self._serialize_preset(
                current_preset,
                include_id=True,
                max_filters=COMPACT_SAVED_PRESET_FILTERS if compact else MAX_SAVED_PRESET_FILTERS,
            ),
            "referenced_preset": referenced_preset,
            "conversation_history": self._serialize_chat_history(chat_history, compact=compact),
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
                    "assistant_message must summarize the tonal intent only, with no per-filter values. "
                    "Do not put reasoning, markdown, comments, or prose outside the JSON object.\n\n"
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
        chat_history: list[dict[str, str]] | None = None,
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
            "referenced_preset": self._serialize_referenced_preset(saved_presets, user_text),
            "conversation_history": self._serialize_chat_history(chat_history),
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

    def shutdown(self) -> None:
        self.clear_context()
        process = self._llama_server_process
        self._llama_server_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_file = self._llama_server_log_file
        self._llama_server_log_file = None
        if log_file is not None and not log_file.closed:
            log_file.close()

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        return AiEqualizerService.is_context_limit_message(str(exc))

    @staticmethod
    def is_context_limit_message(message: str) -> bool:
        message = message.casefold()
        return (
            "exceed context" in message
            or "context window" in message
            or "requested tokens" in message
            or "n_ctx" in message
            or "context limit" in message
        )

    def _serialize_referenced_preset(self, presets: list[Preset] | None, user_text: str) -> dict[str, Any] | None:
        preset = self._find_referenced_preset(presets, user_text)
        if preset is None:
            return None
        return preset.to_dict(include_id=True)

    @staticmethod
    def _serialize_preset(preset: Preset, *, include_id: bool, max_filters: int) -> dict[str, Any]:
        data = preset.to_dict(include_id=include_id)
        data["filters"] = data["filters"][:max_filters]
        data["filters_truncated"] = len(preset.filters) > max_filters
        return data

    @staticmethod
    def _find_referenced_preset(presets: list[Preset] | None, user_text: str) -> Preset | None:
        if not presets:
            return None
        lowered_text = user_text.casefold()
        for preset in presets:
            name = preset.name.strip()
            if name and name.casefold() in lowered_text:
                return preset
        return None

    @staticmethod
    def _serialize_chat_history(
        chat_history: list[dict[str, str]] | None,
        *,
        compact: bool = False,
    ) -> list[dict[str, str]]:
        if not chat_history:
            return []
        items = chat_history[-COMPACT_CHAT_MESSAGES:] if compact else chat_history
        serialized: list[dict[str, str]] = []
        for item in items:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            serialized.append({"role": role, "content": content})
        return serialized

    def _serialize_device_curve(self, curve: FrequencyCurve | None, *, compact: bool = False) -> dict[str, Any]:
        if curve is None or (curve.name == "Default" and curve.path is None):
            return {
                "name": "Default",
                "path": "",
                "raw_txt": "Используются встроенные в ноутбук/монитор динамики.",
                "raw_txt_is_sampled": False,
                "points": [],
            }
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
        if not compact:
            return text.strip(), False
        max_lines = COMPACT_DEVICE_RAW_LINES
        max_chars = COMPACT_DEVICE_RAW_CHARS
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

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().casefold() not in {"0", "false", "no", "off", "нет"}
