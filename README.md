# AIEQ

AIEQ - Windows-прототип параметрического эквалайзера на Python/PySide6 с real-time обработкой через `sounddevice` и VB-Cable.

## Возможности

- график АЧХ с текущим пресетом, сохраненными пресетами для сравнения и АЧХ выбранного устройства;
- фильтры `peaking`, `low_shelf`, `high_shelf`, `low_pass`, `high_pass`, `band_pass`, `notch`;
- огибающий режим сведения полос: пересекающиеся фильтры не суммируются каскадом;
- импорт/экспорт JSON-пресетов и SQLite-хранилище пресетов в `%APPDATA%\AIEQ`;
- AI-чат через локальную GGUF-модель и `llama-cpp-python`;
- вкладка AutoEQ с официальным backend пакета `autoeq` на Python 3.11 и локальным fallback.

## Установка среды

Официальный AutoEq 4.1.2 требует Python `>=3.8,<3.12`, поэтому проект синхронизируется на Python 3.11:

```powershell
uv sync --python 3.11 --extra dev --extra autoeq --extra build --extra ai
Copy-Item .env.example .env
```

## CUDA / GPU

Recommended GPU path is an external `llama.cpp` runtime with the CUDA build of
`llama-server.exe`. Put the executable and all DLLs from the main archive here:

```text
runtime/llama.cpp/llama-server.exe
```

For NVIDIA acceleration, also unpack the separate llama.cpp CUDA DLL archive into
the same folder. The folder must contain `ggml-cuda.dll` and CUDA runtime DLLs
such as `cudart64_*.dll`, `cublas64_*.dll`, and `cublasLt64_*.dll`.

Check that the runtime sees the discrete GPU:

```powershell
.\runtime\llama.cpp\llama-server.exe --list-devices
```

The output must include `CUDA0` / NVIDIA. If it lists only CPU backends, the CUDA
DLL pack is missing or cannot be loaded.

Then keep `.env` in `auto` mode. AIEQ will try `llama-server` first. With the
settings below it will not silently fall back to CPU inference.

```ini
AIEQ_AI_PROVIDER=auto
AIEQ_AI_ALLOW_CPU_FALLBACK=0
AIEQ_LLAMA_SERVER_PATH=runtime/llama.cpp/llama-server.exe
AIEQ_LLAMA_SERVER_HOST=127.0.0.1
AIEQ_LLAMA_SERVER_PORT=8080
AIEQ_LLAMA_SERVER_AUTO_START=1
AIEQ_LLAMA_SERVER_DEVICE=CUDA0
AIEQ_LLAMA_SERVER_REQUIRE_GPU=1
AIEQ_LLAMA_SERVER_PARALLEL=1
AIEQ_LLAMA_SERVER_FLASH_ATTN=on
AIEQ_LLAMA_SERVER_REASONING=auto
AIEQ_LLAMA_SERVER_REASONING_FORMAT=deepseek
AIEQ_LLAMA_SERVER_REASONING_BUDGET=1024
AIEQ_LLAMA_SERVER_CACHE_RAM=0
AIEQ_LLAMA_SERVER_LOG_PATH=
```

When AIEQ starts `llama-server` itself, logs are written to
`%APPDATA%\AIEQ\logs\llama-server.log`. Set `AIEQ_LLAMA_SERVER_LOG_PATH` to use a
different file. If you start the server manually, its logs stay in that terminal.

The older in-process `llama-cpp-python` path is still supported as a CPU fallback. If you specifically want CUDA through the Python package, build it from source:

1. Установите NVIDIA Driver, CUDA Toolkit и Visual Studio Build Tools с C++ workload.
2. Откройте Developer PowerShell for VS.
3. Переустановите backend:

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE="1"
uv pip install --reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
```

GPU offload управляется из `.env`:

```ini
AIEQ_LLAMA_N_GPU_LAYERS=-1
AIEQ_LLAMA_N_BATCH=1024
AIEQ_LLAMA_SERVER_REASONING=auto
```

`-1` означает попытку выгрузить на GPU все слои. Если 4 ГБ VRAM не хватит, поставьте меньшее значение, например `20` или `28`.

## Модель

Рекомендуемая компактная модель: `Qwen/Qwen3-4B-GGUF`, файл `Qwen3-4B-Q4_K_M.gguf`.

1. Создайте папку `models`, если ее нет.
2. Скачайте `Qwen3-4B-Q4_K_M.gguf` со страницы Hugging Face `Qwen/Qwen3-4B-GGUF`.
3. Положите файл сюда:

```text
models/Qwen3-4B-Q4_K_M.gguf
```

4. Проверьте `.env`:

```ini
AIEQ_AI_PROVIDER=auto
AIEQ_LLAMA_MODEL_PATH=models/Qwen3-4B-Q4_K_M.gguf
AIEQ_LLAMA_N_CTX=12288
AIEQ_LLAMA_N_THREADS=8
AIEQ_LLAMA_N_GPU_LAYERS=-1
AIEQ_LLAMA_N_BATCH=1024
AIEQ_LLAMA_MAX_TOKENS=2048
AIEQ_LLAMA_TEMPERATURE=0.35
AIEQ_AUDIO_BLOCK_SIZE=0
AIEQ_AUDIO_LATENCY=50ms
AIEQ_LLAMA_SERVER_PATH=runtime/llama.cpp/llama-server.exe
AIEQ_LLAMA_SERVER_DEVICE=CUDA0
AIEQ_LLAMA_SERVER_REQUIRE_GPU=1
AIEQ_LLAMA_SERVER_PARALLEL=1
AIEQ_LLAMA_SERVER_REASONING=auto
AIEQ_LLAMA_SERVER_REASONING_FORMAT=deepseek
AIEQ_LLAMA_SERVER_REASONING_BUDGET=1024
AIEQ_AUTOEQ_BACKEND=auto
```

AI-вкладка не использует ручные шаблоны пресетов: все содержательные изменения АЧХ формируются подключенной моделью.

Если модель не подключена, чат покажет: `Ваш ИИ-агент не подключен`.

## Запуск из исходников

```powershell
uv run python -m source
```

Типичный сценарий с VB-Cable:

1. В Windows выберите `CABLE Input` как системный вывод.
2. В AIEQ выберите вход `CABLE Output`.
3. В AIEQ выберите реальные наушники/колонки как выход.
4. Нажмите `Старт`.

Для меньшей задержки приложение по умолчанию запускает поток с
`AIEQ_AUDIO_LATENCY=50ms` и `AIEQ_AUDIO_BLOCK_SIZE=0`, чтобы попросить у
PortAudio более короткий буфер. Если драйвер откажется открыть такой поток,
приложение автоматически откатится на `low`. Если появится треск, поставьте
`AIEQ_AUDIO_LATENCY=low` или `AIEQ_AUDIO_BLOCK_SIZE=256`.

Если в AIEQ доступна только одна частота дискретизации, проверьте пару устройств
вне приложения:

```powershell
uv run python scripts/check_audio_formats.py --input "CABLE Output" --output "название выхода"
```

Скрипт покажет, какие частоты отдельно принимает вход, отдельно выход и какие
варианты проходят именно как full-duplex пара. Если 192000 есть в Windows, но не
проходит как full-duplex пара, обычно нужно выбрать WASAPI-устройства напрямую и
поставить одинаковый Default Format в свойствах записи/воспроизведения Windows.
В интерфейсе AIEQ доступны форматы NumPy-потока `float32`, `int32` и `int16`;
packed `int24` скрипт показывает только для диагностики PortAudio.

## AutoEQ

По умолчанию `AIEQ_AUTOEQ_BACKEND=auto`: приложение использует официальный пакет `autoeq`, если он доступен, и переключается на локальный `dmitryz1024`, если официальный backend недоступен.
Локальный `dmitryz1024` использует детальный подбор фильтров по снижению ошибки между устройством и target-кривой.

Для строгой проверки официального backend:

```ini
AIEQ_AUTOEQ_BACKEND=official
```

Для принудительного локального backend:

```ini
AIEQ_AUTOEQ_BACKEND=local
```

## Сборка и запуск EXE

```powershell
.\scripts\build_exe.ps1
```

Результат:

```text
dist\AIEQ\AIEQ.exe
```

Для запуска откройте `dist\AIEQ\AIEQ.exe`. GGUF-модель не вшивается в exe; положите папку `models` рядом с exe или оставьте путь к модели в `.env` относительно рабочей папки запуска.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m compileall source
```
