from __future__ import annotations

import re
from dataclasses import dataclass

from .models import EqFilter, Preset


@dataclass(slots=True)
class IntentResult:
    preset: Preset
    assistant_message: str


_NUMBER_RE = r"[-+]?\d+(?:[.,]\d+)?"


def try_build_intent_preset(user_text: str, current_preset: Preset, saved_presets: list[Preset] | None = None) -> IntentResult | None:
    text = _normalize_text(user_text)
    saved_presets = saved_presets or []
    base = _select_base_preset(text, current_preset, saved_presets)

    exact = _build_exact_filter(text)
    if exact is not None:
        preset = base.clone(keep_id=False)
        preset.filters.append(exact)
        return IntentResult(
            preset=preset.sanitized(),
            assistant_message=_summary_for_exact_filter(exact),
        )

    template_filters, summary = _template_filters(text)
    if template_filters:
        preset = base.clone(keep_id=False)
        preset.filters.extend(template_filters)
        return IntentResult(preset=preset.sanitized(), assistant_message=summary)

    return None


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    return lowered.replace("ё", "е").replace("−", "-").replace("–", "-").replace("—", "-")


def _select_base_preset(text: str, current_preset: Preset, saved_presets: list[Preset]) -> Preset:
    for preset in saved_presets:
        name = _normalize_text(preset.name)
        if name and name in text:
            return preset
    for preset in saved_presets:
        tokens = [token for token in re.findall(r"[^\W_]+", _normalize_text(preset.name), flags=re.UNICODE) if len(token) >= 4]
        if tokens and all(token in text for token in tokens[: min(3, len(tokens))]):
            return preset
    return current_preset


def _build_exact_filter(text: str) -> EqFilter | None:
    freq = _extract_frequency_hz(text)
    gain = _extract_gain_db(text)
    has_action = any(word in text for word in _boost_words() | _cut_words() | {"дб", "db"})
    if freq is None or gain is None or not has_action:
        return None

    if any(word in text for word in _cut_words()) and not _has_explicit_sign(text):
        gain = -abs(gain)
    elif any(word in text for word in _boost_words()) and not _has_explicit_sign(text):
        gain = abs(gain)

    q = _extract_q(text) or _q_from_width(text)
    filter_type = _filter_type_from_text(text, freq)
    if filter_type in {"low_pass", "high_pass", "band_pass", "notch"}:
        gain = 0.0
    return EqFilter(filter_type, freq, q, gain).sanitized()


def _extract_frequency_hz(text: str) -> float | None:
    patterns = [
        rf"({_NUMBER_RE})\s*(?:кгц|khz|k hz|килогерц)",
        rf"({_NUMBER_RE})\s*k\b",
        rf"(?:центр(?:ом)?|частот[аеы]?|freq(?:uency)?)\s*(?:в|на|=|:)?\s*({_NUMBER_RE})\s*(?:гц|hz)?",
        rf"(?:в районе|около)\s*({_NUMBER_RE})\s*(?:гц|hz)?",
        rf"({_NUMBER_RE})\s*(?:гц|hz)\b",
    ]
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.UNICODE)
        if not match:
            continue
        value = _to_float(match.group(1))
        if value is None:
            continue
        if index <= 1:
            value *= 1000.0
        return float(value)
    return None


def _extract_gain_db(text: str) -> float | None:
    patterns = [
        rf"([-+]\s*\d+(?:[.,]\d+)?)\s*(?:дб|db)\b",
        rf"(?:на|gain|гейн|усилен(?:ие)?|подъем|срез|убери|убрать|убавь|уменьш(?:и|ить)?|ослабь?)\s*({_NUMBER_RE})\s*(?:дб|db)\b",
        rf"({_NUMBER_RE})\s*(?:дб|db)\s*(?:вверх|подъем|boost|плюс|прибав)",
        rf"({_NUMBER_RE})\s*(?:дб|db)\s*(?:вниз|срез|cut|минус|убав)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.UNICODE)
        if not match:
            continue
        value = _to_float(match.group(1).replace(" ", ""))
        if value is None:
            continue
        return float(value)
    return None


def _extract_q(text: str) -> float | None:
    match = re.search(rf"\bq\s*[:=]?\s*({_NUMBER_RE})", text, flags=re.UNICODE)
    if not match:
        return None
    return _to_float(match.group(1))


def _q_from_width(text: str) -> float:
    if any(word in text for word in {"очень узк", "very narrow", "резонанс", "свист", "писк"}):
        return 6.0
    if any(word in text for word in {"узк", "narrow", "точечн"}):
        return 3.0
    if any(word in text for word in {"широк", "wide", "broad", "плавн", "общ", "мягк"}):
        return 0.7
    return 1.0


def _filter_type_from_text(text: str, freq: float) -> str:
    if any(word in text for word in {"notch", "режектор", "выреж", "резонанс", "свист", "писк"}):
        return "notch"
    if "band pass" in text or "bandpass" in text or "полосовой" in text:
        return "band_pass"
    if "low pass" in text or "lowpass" in text or "фнч" in text:
        return "low_pass"
    if "high pass" in text or "highpass" in text or "фвч" in text:
        return "high_pass"
    if "полк" in text or "shelf" in text:
        if freq < 1000 or any(word in text for word in {"низ", "бас", "low"}):
            return "low_shelf"
        return "high_shelf"
    return "peaking"


def _template_filters(text: str) -> tuple[list[EqFilter], str]:
    if any(word in text for word in {"v-образ", "v образ", "ви образ", "v-shaped", "масс-медиа", "медиа"}):
        return (
            [
                EqFilter("low_shelf", 95, 0.7, 3.0),
                EqFilter("peaking", 450, 0.9, -1.5),
                EqFilter("peaking", 2500, 1.0, 1.2),
                EqFilter("high_shelf", 8500, 0.7, 3.0),
            ],
            "Сделал умеренный V-образный характер: плотнее низ, чуть чище середина и заметнее верх.",
        )
    if any(word in text for word in {"крутой звук", "круче", "энергич", "драйв", "вау", "fun"}):
        return (
            [
                EqFilter("low_shelf", 90, 0.7, 2.2),
                EqFilter("peaking", 280, 0.9, -1.2),
                EqFilter("peaking", 2200, 1.0, 1.4),
                EqFilter("high_shelf", 9000, 0.75, 2.0),
            ],
            "Сделал более энергичную и чистую подачу без радикального перекоса тонального баланса.",
        )
    if any(word in text for word in {"воздух", "ярч", "деталь", "искр", "верх"}):
        return (
            [
                EqFilter("peaking", 3500, 1.2, 1.2),
                EqFilter("high_shelf", 9000, 0.7, 2.0),
            ],
            "Добавил немного присутствия и воздуха, сохранив подъемы в умеренных пределах.",
        )
    if any(word in text for word in {"бас", "низ", "панч", "саб"}):
        return (
            [
                EqFilter("low_shelf", 95, 0.7, 2.5),
                EqFilter("peaking", 250, 0.9, -0.8),
            ],
            "Усилил низ и немного подчистил область, где бас часто становится мутным.",
        )
    if any(word in text for word in {"мут", "гул", "бубн", "короб"}):
        return (
            [
                EqFilter("peaking", 180, 1.1, -2.0),
                EqFilter("peaking", 420, 1.0, -1.4),
            ],
            "Подчистил низкую середину и область гула, чтобы звук стал собраннее.",
        )
    if any(word in text for word in {"резк", "сибил", "шип", "колк", "устал"}):
        return (
            [
                EqFilter("peaking", 3200, 1.4, -1.5),
                EqFilter("peaking", 6500, 2.5, -2.0),
            ],
            "Смягчил резкость и шипящие области, не делая звук темным.",
        )
    return [], ""


def _summary_for_exact_filter(eq_filter: EqFilter) -> str:
    if eq_filter.type == "notch":
        return "Добавил точечное подавление проблемной области и сохранил остальную тональность."
    if eq_filter.gain >= 0:
        return "Добавил запрошенный подъем в выбранной области и сохранил существующую основу пресета."
    return "Добавил запрошенное ослабление в выбранной области и сохранил существующую основу пресета."


def _boost_words() -> set[str]:
    return {"подъем", "поднять", "приподнять", "прибав", "усил", "boost", "вверх", "плюс", "добав"}


def _cut_words() -> set[str]:
    return {"срез", "срезать", "убрать", "убери", "убав", "уменьш", "ослаб", "cut", "вниз", "минус"}


def _has_explicit_sign(text: str) -> bool:
    return bool(re.search(r"[-+]\s*\d+(?:[.,]\d+)?\s*(?:дб|db)", text, flags=re.UNICODE))


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None
