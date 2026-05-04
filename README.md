# AIEQ

AIEQ - прототип параметрического эквалайзера под Windows на Python.

Возможности:

- отображение текущей АЧХ и сохраненных пресетов разными цветами;
- параметрические фильтры `peaking`, `low_shelf`, `high_shelf`, `low_pass`, `high_pass`, `band_pass`, `notch`;
- огибающий режим сведения полос: пересекающиеся фильтры не каскадируются, итоговая АЧХ выбирает самое выраженное отклонение в каждой точке;
- импорт и экспорт пресетов в JSON;
- SQLite-память пресетов для быстрого сравнения;
- выбор входного и выходного аудиоустройства с подписью аудио-подсистемы в квадратных скобках;
- real-time обработка через `sounddevice` и VB-Cable;
- AI-чат через локальную Ollama или OpenAI, который превращает пожелание по звуку в новый пресет, сохраняет его, применяет и показывает на графике.
- сохранение размеров окна и ширины секции AI-чата между запусками.

## Установка

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Скопируйте `.env.example` в `.env` и настройте AI-провайдер:

```powershell
Copy-Item .env.example .env
```

Для бесплатного локального режима с Ollama:

```ini
AIEQ_AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=300
```

Если `OLLAMA_MODEL` оставить пустым, приложение попробует взять первую установленную модель из Ollama. Если AI-модель не подключена, чат покажет системное сообщение: `Ваш ИИ-агент не подключен`.

Для OpenAI-режима:

```ini
AIEQ_AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
AIEQ_OPENAI_MODEL=gpt-5.2
```

## Запуск

```powershell
python -m aieq
```

Типичный сценарий с VB-Cable:

1. В Windows выберите `CABLE Input` как системный выход.
2. В AIEQ выберите вход `CABLE Output`.
3. В AIEQ выберите реальные наушники/колонки как выход.
4. Нажмите `Старт`.

## Пресет JSON

```json
{
  "version": 1,
  "name": "Example",
  "filters": [
    { "type": "low_shelf", "freq": 90, "q": 0.7, "gain": 3.0 },
    { "type": "peaking", "freq": 350, "q": 1.1, "gain": -2.0 }
  ]
}
```

## Проверки

```powershell
pytest
python -m compileall aieq
```
