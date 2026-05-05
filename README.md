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

Код приложения теперь передает в `llama-cpp-python` параметры GPU offload (`n_gpu_layers`, `n_batch`). На Windows для реального CUDA-ускорения надежный вариант - собрать `llama-cpp-python` из исходников с `GGML_CUDA=on`.

Минимальная подготовка:

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
AIEQ_LLAMA_N_BATCH=512
```

`-1` означает попытку выгрузить на GPU все слои. Если 4 ГБ VRAM не хватит, поставьте меньшее значение, например `20` или `28`.

## Модель

Рекомендуемая компактная модель: `Qwen/Qwen2.5-3B-Instruct-GGUF`, файл `qwen2.5-3b-instruct-q4_k_m.gguf`.

1. Создайте папку `models`, если ее нет.
2. Скачайте `qwen2.5-3b-instruct-q4_k_m.gguf` со страницы Hugging Face `Qwen/Qwen2.5-3B-Instruct-GGUF`.
3. Положите файл сюда:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

4. Проверьте `.env`:

```ini
AIEQ_AI_PROVIDER=llama_cpp
AIEQ_LLAMA_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf
AIEQ_LLAMA_N_CTX=8192
AIEQ_LLAMA_N_THREADS=7
AIEQ_LLAMA_N_GPU_LAYERS=-1
AIEQ_LLAMA_N_BATCH=512
AIEQ_AUTOEQ_BACKEND=auto
```

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

## AutoEQ

По умолчанию `AIEQ_AUTOEQ_BACKEND=auto`: приложение использует официальный пакет `autoeq`, если он доступен, и переключается на локальный fallback только если официальный backend недоступен.

Для строгой проверки официального backend:

```ini
AIEQ_AUTOEQ_BACKEND=official
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
