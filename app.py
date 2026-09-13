"""Flask API для многодиалогового DeepSeek Agent."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from agent import (
    DEEPSEEK_MODELS,
    PROVIDER_CAPABILITIES,
    Agent,
    AgentSettings,
    DeepSeekProvider,
    dialogue_token_totals,
    normalize_token_usage,
)
from context_manager import (
    active_path_messages,
    add_fact,
    append_tree_message,
    begin_branch,
    branch_points,
    delete_fact,
    edit_fact,
    ensure_context_management,
    facts_token_totals,
    request_history,
    select_branch,
    set_context_mode,
    set_window_sizes,
    summary_token_totals,
    update_sticky_facts,
    update_summaries,
)
from presets import PresetManager
from storage import JsonStorage, now_iso
from voice import WhisperService


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
MAX_MESSAGE_LENGTH = 50_000


def create_app(
    data_dir: Path | None = None,
    agent_instance: Agent | None = None,
    voice_service: WhisperService | None = None,
) -> Flask:
    flask_app = Flask(__name__)
    flask_app.json.ensure_ascii = False
    flask_app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024
    storage = JsonStorage(data_dir or BASE_DIR / "data")
    preset_manager = PresetManager(storage)
    chat_agent = agent_instance or Agent(DeepSeekProvider())
    whisper = voice_service or WhisperService(BASE_DIR, autostart=data_dir is None)

    flask_app.extensions["json_storage"] = storage
    flask_app.extensions["preset_manager"] = preset_manager
    flask_app.extensions["chat_agent"] = chat_agent
    flask_app.extensions["whisper_service"] = whisper

    @flask_app.get("/")
    def index():
        return render_template("index.html")

    @flask_app.get("/api/state")
    def state():
        default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if default_model not in DEEPSEEK_MODELS:
            default_model = "deepseek-v4-flash"
        return jsonify({
            "ok": True,
            "provider": PROVIDER_CAPABILITIES,
            "default_settings": AgentSettings(model=default_model).to_dict(),
            "conversations": storage.list_conversations(),
            "presets": preset_manager.list(),
        })

    @flask_app.get("/api/voice/status")
    def voice_status():
        return jsonify({"ok": True, "voice": whisper.status()})

    @flask_app.post("/api/voice/start")
    def voice_start():
        whisper.start()
        return jsonify({"ok": True, "voice": whisper.status()}), 202

    @flask_app.post("/api/voice/transcribe")
    def voice_transcribe():
        uploaded = request.files.get("audio")
        if uploaded is None:
            return api_error("Аудиозапись не передана.", 400)
        try:
            result = whisper.transcribe(
                uploaded.read(),
                uploaded.filename or "recording.webm",
                uploaded.mimetype or "application/octet-stream",
            )
            return jsonify({"ok": True, **result})
        except ValueError as error:
            return api_error(str(error), 400)
        except RuntimeError as error:
            return api_error(str(error), 503)

    @flask_app.post("/api/conversations")
    def create_conversation():
        try:
            conversation = storage.create_conversation(json_body().get("title", "Новый диалог"))
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str):
        try:
            return jsonify({"ok": True, "conversation": with_token_totals(storage.get_conversation(conversation_id))})
        except (ValueError, FileNotFoundError):
            return api_error("Диалог не найден.", 404)

    @flask_app.patch("/api/conversations/<conversation_id>")
    def rename_conversation(conversation_id: str):
        try:
            conversation = storage.rename_conversation(conversation_id, json_body().get("title", ""))
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.patch("/api/conversations/<conversation_id>/context")
    def update_conversation_context(conversation_id: str):
        try:
            conversation = storage.get_conversation(conversation_id)
            previous_mode = ensure_context_management(conversation)["mode"]
            data = json_body()
            mode = data.get("mode")
            set_context_mode(conversation, mode)
            set_window_sizes(
                conversation,
                sliding=data.get("sliding_window_exchanges"),
                facts=data.get("facts_window_exchanges"),
            )
            if previous_mode != mode:
                labels = {
                    "full": "Полная история",
                    "summary": "Summary + последние обмены",
                    "sliding": "Sliding Window",
                    "facts": "Sticky Facts",
                    "branching": "Branching",
                }
                conversation["messages"].append({
                    "id": uuid.uuid4().hex,
                    "role": "event",
                    "parent_id": conversation["context_management"].get("active_leaf_id"),
                    "content": f"Режим контекста: {labels[mode]}.",
                    "created_at": now_iso(),
                })
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str):
        try:
            storage.delete_conversation(conversation_id)
            return jsonify({"ok": True})
        except (ValueError, FileNotFoundError):
            return api_error("Диалог не найден.", 404)

    @flask_app.post("/api/conversations/<conversation_id>/messages")
    def send_message(conversation_id: str):
        try:
            data = json_body()
            content = require_message(data.get("content"))
            settings = AgentSettings.from_dict(data.get("settings"))
            source = configuration_source(data.get("preset_id"), settings, preset_manager)
            conversation = storage.get_conversation(conversation_id)
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)

        exchange_id = uuid.uuid4().hex
        created_at = now_iso()
        settings_snapshot = settings.to_dict()
        ensure_context_management(conversation)
        previous_source = last_configuration(active_path_messages(conversation))
        if previous_source is not None and (
            previous_source.get("settings") != settings_snapshot
            or previous_source.get("configuration_source") != source
        ):
            conversation["messages"].append({
                "id": uuid.uuid4().hex,
                "role": "event",
                "parent_id": conversation["context_management"].get("active_leaf_id"),
                "content": configuration_event_text(source),
                "created_at": created_at,
            })

        user_message = {
            "id": uuid.uuid4().hex,
            "exchange_id": exchange_id,
            "role": "user",
            "content": content,
            "created_at": created_at,
            "technical": {
                "request_status": "pending",
                "configuration_source": source,
                "settings": settings_snapshot,
            },
        }
        append_tree_message(conversation, user_message)
        if conversation.get("title") == "Новый диалог":
            conversation["title"] = content.replace("\n", " ")[:60]
        conversation = storage.save_conversation(conversation)

        try:
            update_sticky_facts(conversation, chat_agent, content)
            update_summaries(conversation, chat_agent)
            history_for_request, active_summary_text, active_facts, context_snapshot = request_history(conversation)
            conversation = storage.save_conversation(conversation)
            result = chat_agent.reply(
                history_for_request,
                content,
                settings,
                summary=active_summary_text,
                facts=active_facts,
            )
            usage = normalize_token_usage(result.technical.get("usage"))
            for message in conversation["messages"]:
                if message.get("id") == user_message["id"]:
                    message["technical"].update({
                        "request_status": "completed",
                        "token_usage": {
                            "context_tokens": usage["input_tokens"],
                            "cached_context_tokens": usage["cached_input_tokens"],
                            "uncached_context_tokens": usage["uncached_input_tokens"],
                        },
                        "context_management": context_snapshot,
                    })
                    break
            assistant_message = {
                "id": uuid.uuid4().hex,
                "exchange_id": exchange_id,
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": "completed",
                    "configuration_source": source,
                    "settings": settings_snapshot,
                    "context_management": context_snapshot,
                },
            }
            append_tree_message(conversation, assistant_message, parent_id=user_message["id"])
            totals = dialogue_token_totals(conversation["messages"])
            conversation["token_totals"] = totals
            assistant_message["technical"]["dialogue_totals"] = totals
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except Exception as error:
            flask_app.logger.exception("Ошибка DeepSeek API")
            message = friendly_api_error(error)
            for item in conversation["messages"]:
                if item.get("id") == user_message["id"]:
                    item["technical"].update({"request_status": "failed", "error": message})
                    break
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": False, "error": message, "conversation": with_token_totals(conversation)}), 502

    @flask_app.get("/api/conversations/<conversation_id>/export")
    def export_conversation(conversation_id: str):
        try:
            bundle = storage.export_bundle(conversation_id)
            bundle["conversation"] = with_token_totals(bundle["conversation"])
        except (ValueError, FileNotFoundError):
            return api_error("Диалог не найден.", 404)
        payload = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(
            BytesIO(payload),
            mimetype="application/json; charset=utf-8",
            as_attachment=True,
            download_name=f"deepseek-dialog-{conversation_id[:8]}.json",
        )

    @flask_app.post("/api/conversations/<conversation_id>/branches")
    def create_branch(conversation_id: str):
        """Начинает ветку после сообщения; после user сразу генерирует новый ответ."""
        try:
            conversation = storage.get_conversation(conversation_id)
            if ensure_context_management(conversation)["mode"] != "branching":
                raise ValueError("Сначала выберите стратегию Branching.")
            checkpoint = begin_branch(conversation, json_body().get("checkpoint_id"))
            if checkpoint["role"] == "assistant":
                conversation = storage.save_conversation(conversation)
                return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201

            technical = checkpoint.get("technical", {})
            settings = AgentSettings.from_dict(technical.get("settings"))
            source = technical.get("configuration_source") or {
                "type": "custom", "preset_id": None, "preset_name": None,
            }
            history, summary, facts, snapshot = request_history(conversation)
            result = chat_agent.reply(
                history,
                checkpoint["content"],
                settings,
                summary=summary,
                facts=facts,
            )
            assistant = {
                "id": uuid.uuid4().hex,
                "exchange_id": checkpoint.get("exchange_id"),
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
                "created_at": now_iso(),
                "technical": {
                    **result.technical,
                    "request_status": "completed",
                    "configuration_source": source,
                    "settings": settings.to_dict(),
                    "context_management": snapshot,
                    "regenerated_from_checkpoint": checkpoint["id"],
                },
            }
            append_tree_message(conversation, assistant, parent_id=checkpoint["id"])
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except (ValueError, KeyError) as error:
            return api_error(str(error).strip("'"), 400)
        except Exception as error:
            flask_app.logger.exception("Ошибка DeepSeek API при создании ветки")
            return api_error(friendly_api_error(error), 502)

    @flask_app.patch("/api/conversations/<conversation_id>/branches/active")
    def change_active_branch(conversation_id: str):
        try:
            conversation = storage.get_conversation(conversation_id)
            data = json_body()
            select_branch(conversation, data.get("checkpoint_id"), data.get("child_id"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.post("/api/conversations/<conversation_id>/facts")
    def create_manual_fact(conversation_id: str):
        try:
            conversation = storage.get_conversation(conversation_id)
            data = json_body()
            fact = add_fact(conversation, data.get("key"), data.get("value"))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)}), 201
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/conversations/<conversation_id>/facts/<fact_id>")
    def update_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = storage.get_conversation(conversation_id)
            data = json_body()
            fact = edit_fact(conversation, fact_id, data.get("key"), data.get("value"), data.get("locked", True))
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "fact": fact, "conversation": with_token_totals(conversation)})
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/conversations/<conversation_id>/facts/<fact_id>")
    def remove_manual_fact(conversation_id: str, fact_id: str):
        try:
            conversation = storage.get_conversation(conversation_id)
            delete_fact(conversation, fact_id)
            conversation = storage.save_conversation(conversation)
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)})
        except FileNotFoundError:
            return api_error("Диалог не найден.", 404)
        except KeyError:
            return api_error("Факт не найден.", 404)

    @flask_app.post("/api/import")
    def import_conversation():
        try:
            conversation = storage.import_bundle(json_body())
            return jsonify({"ok": True, "conversation": with_token_totals(conversation)}), 201
        except (ValueError, OSError, json.JSONDecodeError) as error:
            return api_error(f"Не удалось импортировать файл: {error}", 400)

    @flask_app.post("/api/presets")
    def create_preset():
        try:
            return jsonify({"ok": True, "preset": preset_manager.create(json_body())}), 201
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.put("/api/presets/<preset_id>")
    def update_preset(preset_id: str):
        try:
            return jsonify({"ok": True, "preset": preset_manager.update(preset_id, json_body())})
        except KeyError:
            return api_error("Пресет не найден.", 404)
        except ValueError as error:
            return api_error(str(error), 400)

    @flask_app.delete("/api/presets/<preset_id>")
    def delete_preset(preset_id: str):
        try:
            preset_manager.delete(preset_id)
            return jsonify({"ok": True})
        except KeyError:
            return api_error("Пресет не найден.", 404)

    return flask_app


def json_body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект.")
    return value


def require_message(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Введите сообщение.")
    content = value.strip()
    if len(content) > MAX_MESSAGE_LENGTH:
        raise ValueError("Сообщение длиннее 50 000 символов.")
    return content


def configuration_source(preset_id: Any, settings: AgentSettings, manager: PresetManager) -> dict[str, Any]:
    if not preset_id:
        return {"type": "custom", "preset_id": None, "preset_name": None}
    preset = manager.get(str(preset_id))
    source_type = "preset" if preset["settings"] == settings.to_dict() else "preset_modified"
    return {"type": source_type, "preset_id": preset["id"], "preset_name": preset["name"]}


def last_configuration(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("technical"), dict):
            return message["technical"]
    return None


def configuration_event_text(source: dict[str, Any]) -> str:
    if source["type"] == "preset":
        return f"Выбран пресет «{source['preset_name']}»."
    if source["type"] == "preset_modified":
        return f"Настройки пресета «{source['preset_name']}» изменены для следующего запроса."
    return "Для следующего запроса выбраны разовые настройки."


def api_error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def with_token_totals(conversation: dict[str, Any]) -> dict[str, Any]:
    """Добавляет вычисляемые представления, не изменяя сохранённый оригинал."""
    result = deepcopy(conversation)
    ensure_context_management(result)
    result["token_totals"] = dialogue_token_totals(result.get("messages", []))
    visible = active_path_messages(result)
    result["visible_messages"] = visible
    result["active_path_token_totals"] = dialogue_token_totals(visible)
    result["branch_points"] = branch_points(result)
    result["summary_token_totals"] = summary_token_totals(result.get("summaries", []))
    result["facts_token_totals"] = facts_token_totals(result.get("fact_revisions", []))
    return result


def friendly_api_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    context_markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
    )
    if any(marker in lowered for marker in context_markers):
        return (
            "Контекст диалога превысил лимит модели. "
            "Создайте новый диалог или сократите историю и повторите запрос."
        )
    if "api key" in lowered or "authentication" in lowered or "401" in lowered:
        return "DeepSeek отклонил API-ключ. Проверьте DEEPSEEK_API_KEY в файле .env."
    if "429" in lowered or "rate limit" in lowered:
        return "DeepSeek временно ограничил количество запросов. Попробуйте немного позже."
    if "timeout" in lowered or "timed out" in lowered:
        return "DeepSeek не успел ответить. Повторите запрос."
    if "connection" in lowered or "connect" in lowered or "network" in lowered:
        return "Не удалось соединиться с DeepSeek. Проверьте интернет, VPN или прокси."
    return f"Не удалось получить ответ DeepSeek: {text}"


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
