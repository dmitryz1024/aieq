from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FILTER_TYPES: tuple[str, ...] = (
    "peaking",
    "low_shelf",
    "high_shelf",
    "low_pass",
    "high_pass",
    "band_pass",
    "notch",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class EqFilter:
    type: str = "peaking"
    freq: float = 1000.0
    q: float = 1.0
    gain: float = 0.0
    enabled: bool = True

    def sanitized(self, sample_rate: float = 48000.0) -> "EqFilter":
        max_freq = min(20000.0, sample_rate * 0.475)
        filter_type = self.type if self.type in FILTER_TYPES else "peaking"
        return EqFilter(
            type=filter_type,
            freq=clamp(float(self.freq), 20.0, max_freq),
            q=clamp(float(self.q), 0.1, 18.0),
            gain=clamp(float(self.gain), -24.0, 24.0),
            enabled=bool(self.enabled),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "freq": round(float(self.freq), 4),
            "q": round(float(self.q), 4),
            "gain": round(float(self.gain), 4),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EqFilter":
        return cls(
            type=str(data.get("type", "peaking")),
            freq=float(data.get("freq", 1000.0)),
            q=float(data.get("q", 1.0)),
            gain=float(data.get("gain", 0.0)),
            enabled=bool(data.get("enabled", True)),
        ).sanitized()


NEW_PRESET_NAME = "New"
PRESET_NAME_MAX_LENGTH = 180


@dataclass(slots=True)
class Preset:
    name: str = NEW_PRESET_NAME
    filters: list[EqFilter] = field(default_factory=list)
    id: int | None = None
    version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def clone(self, *, name: str | None = None, keep_id: bool = False) -> "Preset":
        return Preset(
            name=name if name is not None else self.name,
            filters=[EqFilter.from_dict(item.to_dict()) for item in self.filters],
            id=self.id if keep_id else None,
            version=self.version,
            created_at=self.created_at,
            updated_at=utc_now_iso(),
        )

    def sanitized(self, sample_rate: float = 48000.0) -> "Preset":
        clean = self.clone(keep_id=True)
        clean.name = (clean.name or "Untitled").strip()[:PRESET_NAME_MAX_LENGTH]
        clean.filters = [item.sanitized(sample_rate) for item in clean.filters]
        clean.updated_at = utc_now_iso()
        return clean

    def to_dict(self, *, include_id: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "filters": [item.to_dict() for item in self.filters],
        }
        if include_id and self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preset":
        filters = data.get("filters", [])
        if not isinstance(filters, list):
            filters = []
        preset = cls(
            name=str(data.get("name", "Imported preset")).strip()[:PRESET_NAME_MAX_LENGTH] or "Imported preset",
            filters=[EqFilter.from_dict(item) for item in filters if isinstance(item, dict)],
            id=int(data["id"]) if data.get("id") is not None else None,
            version=int(data.get("version", 1)),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
        )
        return preset.sanitized()


def flat_preset() -> Preset:
    return Preset(name=NEW_PRESET_NAME, filters=[])
