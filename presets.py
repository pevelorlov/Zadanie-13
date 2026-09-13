"""Управление переносимыми пресетами агента."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

from agent import AgentSettings
from storage import JsonStorage, now_iso


class PresetManager:
    def __init__(self, storage: JsonStorage) -> None:
        self.storage = storage

    def list(self) -> list[dict[str, Any]]:
        return self.storage.load_presets_document()["presets"]

    def get(self, preset_id: str) -> dict[str, Any]:
        for preset in self.list():
            if preset.get("id") == preset_id:
                return preset
        raise KeyError("Пресет не найден.")

    def create(self, value: Any) -> dict[str, Any]:
        preset = self._validated(value, new_id=uuid.uuid4().hex)
        document = self.storage.load_presets_document()
        document["presets"].append(preset)
        self.storage.save_presets_document(document)
        return deepcopy(preset)

    def update(self, preset_id: str, value: Any) -> dict[str, Any]:
        document = self.storage.load_presets_document()
        for index, current in enumerate(document["presets"]):
            if current.get("id") == preset_id:
                preset = self._validated(value, new_id=preset_id, created_at=current.get("created_at"))
                document["presets"][index] = preset
                self.storage.save_presets_document(document)
                return deepcopy(preset)
        raise KeyError("Пресет не найден.")

    def delete(self, preset_id: str) -> None:
        document = self.storage.load_presets_document()
        kept = [item for item in document["presets"] if item.get("id") != preset_id]
        if len(kept) == len(document["presets"]):
            raise KeyError("Пресет не найден.")
        document["presets"] = kept
        self.storage.save_presets_document(document)

    def merge_imported(self, values: list[Any]) -> None:
        document = self.storage.load_presets_document()
        existing = {item.get("id") for item in document["presets"]}
        for raw in values:
            if not isinstance(raw, dict):
                continue
            original_id = str(raw.get("id", ""))
            if original_id in existing:
                continue
            try:
                preset = self._validated(raw, new_id=original_id or uuid.uuid4().hex)
            except ValueError:
                continue
            document["presets"].append(preset)
            existing.add(preset["id"])
        self.storage.save_presets_document(document)

    @staticmethod
    def _validated(value: Any, new_id: str, created_at: str | None = None) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Пресет должен быть JSON-объектом.")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("Введите название пресета.")
        if len(name) > 80:
            raise ValueError("Название пресета длиннее 80 символов.")
        description = str(value.get("description", "")).strip()[:500]
        settings = AgentSettings.from_dict(value.get("settings", {})).to_dict()
        timestamp = now_iso()
        return {
            "id": new_id,
            "name": name,
            "description": description,
            "created_at": created_at or timestamp,
            "updated_at": timestamp,
            "settings": settings,
        }
