"""Потокобезопасное JSON-хранилище диалогов и переносимых пакетов."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JsonStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.conversations_dir = data_dir / "conversations"
        self.presets_path = data_dir / "presets.json"
        self._lock = threading.RLock()
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        if not self.presets_path.exists():
            self._write_json(self.presets_path, {"schema_version": 1, "presets": []})

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for path in self.conversations_dir.glob("*.json"):
                try:
                    item = self._read_json(path)
                    result.append(
                        {
                            "id": item["id"],
                            "title": item.get("title", "Новый диалог"),
                            "created_at": item.get("created_at", ""),
                            "updated_at": item.get("updated_at", ""),
                            "message_count": sum(
                                1 for message in item.get("messages", [])
                                if message.get("role") in {"user", "assistant"}
                            ),
                        }
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def create_conversation(self, title: str = "Новый диалог") -> dict[str, Any]:
        title = clean_title(title)
        timestamp = now_iso()
        conversation = {
            "schema_version": 3,
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
            "context_management": {
                "mode": "full",
                "keep_recent_exchanges": 5,
                "summary_batch_exchanges": 5,
                "summary_model": "deepseek-v4-flash",
                "summary_reasoning_enabled": False,
                "active_summary_id": None,
                "sliding_window_exchanges": 5,
                "facts_window_exchanges": 5,
                "active_leaf_id": None,
                "pending_branch_from_id": None,
            },
            "summaries": [],
            "facts": [],
            "fact_revisions": [],
        }
        with self._lock:
            self._write_json(self._conversation_path(conversation["id"]), conversation)
        return deepcopy(conversation)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._read_json(self._conversation_path(conversation_id)))

    def save_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(conversation.get("id", ""))
        path = self._conversation_path(conversation_id)
        saved = deepcopy(conversation)
        saved["schema_version"] = 3
        saved["updated_at"] = now_iso()
        with self._lock:
            self._write_json(path, saved)
        return deepcopy(saved)

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        conversation["title"] = clean_title(title)
        return self.save_conversation(conversation)

    def delete_conversation(self, conversation_id: str) -> None:
        path = self._conversation_path(conversation_id)
        with self._lock:
            path.unlink()

    def load_presets_document(self) -> dict[str, Any]:
        with self._lock:
            value = self._read_json(self.presets_path)
        if not isinstance(value.get("presets"), list):
            raise ValueError("Файл пресетов повреждён.")
        return deepcopy(value)

    def save_presets_document(self, document: dict[str, Any]) -> None:
        with self._lock:
            self._write_json(self.presets_path, document)

    def export_bundle(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        library = {
            preset.get("id"): preset for preset in self.load_presets_document()["presets"]
        }
        snapshots: dict[str, dict[str, Any]] = {}
        for message in conversation.get("messages", []):
            technical = message.get("technical", {})
            source = technical.get("configuration_source", {})
            preset_id = source.get("preset_id")
            if message.get("role") != "user" or not preset_id or preset_id in snapshots:
                continue
            snapshots[preset_id] = deepcopy(library.get(preset_id) or {
                "id": preset_id,
                "name": source.get("preset_name") or "Импортированный пресет",
                "description": "Восстановлен из снимка настроек диалога.",
                "created_at": message.get("created_at", now_iso()),
                "updated_at": message.get("created_at", now_iso()),
                "settings": technical.get("settings", {}),
            })
        return {
            "format": "deepseek-agent-export",
            "schema_version": 1,
            "exported_at": now_iso(),
            "conversation": conversation,
            "presets": list(snapshots.values()),
        }

    def import_bundle(self, bundle: Any) -> dict[str, Any]:
        if not isinstance(bundle, dict) or bundle.get("format") != "deepseek-agent-export":
            raise ValueError("Это не файл экспорта DeepSeek Agent.")
        source = bundle.get("conversation")
        if not isinstance(source, dict) or not isinstance(source.get("messages"), list):
            raise ValueError("В файле отсутствует корректный диалог.")

        imported = deepcopy(source)
        imported["id"] = uuid.uuid4().hex
        imported["title"] = clean_title(f"{source.get('title', 'Импортированный диалог')} — импорт")
        imported["created_at"] = now_iso()
        imported["updated_at"] = imported["created_at"]
        imported["schema_version"] = 3

        from agent import AgentSettings
        for message in imported["messages"]:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "event"}:
                raise ValueError("В импортируемой истории найдено некорректное сообщение.")
            technical = message.get("technical", {})
            if message.get("role") in {"user", "assistant"}:
                AgentSettings.from_dict(technical.get("settings", {}))

        self.save_conversation(imported)

        raw_presets = bundle.get("presets", [])
        if isinstance(raw_presets, list):
            from presets import PresetManager
            PresetManager(self).merge_imported(raw_presets)
        return imported

    def _conversation_path(self, conversation_id: str) -> Path:
        if not ID_PATTERN.fullmatch(conversation_id):
            raise ValueError("Некорректный идентификатор диалога.")
        return self.conversations_dir / f"{conversation_id}.json"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Ожидался JSON-объект.")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def clean_title(value: Any) -> str:
    title = str(value).strip()
    if not title:
        return "Новый диалог"
    return title[:120]
