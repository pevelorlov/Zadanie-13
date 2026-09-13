"""Локальные тесты без реальных запросов и расходов DeepSeek API."""

import json
import socket
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from agent import Agent, AgentResult, AgentSettings, DeepSeekProvider, dialogue_token_totals
from app import create_app, friendly_api_error
from context_manager import completed_exchanges, ensure_context_management, summary_token_totals
from main import find_available_port
from presets import PresetManager
from storage import JsonStorage


class FakeProvider:
    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def complete(self, messages, settings):
        self.calls.append((messages, settings))
        if self.fail:
            raise RuntimeError("Connection error")
        content = (
            '{"facts":[{"key":"preferences.language","value":"Русский"}]}'
            if settings.response_format == "json_object"
            else "Тестовый ответ"
        )
        return AgentResult(
            content=content,
            reasoning_content="Скрытое рассуждение",
            technical={
                "provider": "deepseek",
                "request_id": "req-test",
                "model": settings.model,
                "finish_reason": "stop",
                "elapsed_seconds": 0.321,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "cached_input_tokens": 7,
                    "uncached_input_tokens": 3,
                    "reasoning_tokens": 4,
                },
                "settings": settings.to_dict(),
            },
        )


class FakeVoiceService:
    def __init__(self):
        self.started = False
        self.received = None

    def status(self):
        return {
            "phase": "ready", "ready": True, "message": "Whisper готов · тест",
            "model": "large-v3-turbo", "backend": "Vulkan", "language": "ru",
        }

    def start(self):
        self.started = True

    def transcribe(self, audio, filename, content_type):
        self.received = (audio, filename, content_type)
        return {"text": "Распознанный текст", "language": "ru"}

    def stop(self):
        pass


class AgentTests(unittest.TestCase):
    def test_agent_owns_context_and_uses_current_system_prompt(self):
        provider = FakeProvider()
        agent = Agent(provider)
        settings = AgentSettings(system_prompt="Ты технический специалист", temperature=0.2)
        history = [
            {"role": "user", "content": "Меня зовут Павел", "technical": {"request_status": "completed"}},
            {"role": "assistant", "content": "Запомнил", "technical": {"request_status": "completed"}},
            {"role": "event", "content": "Смена настроек"},
        ]
        agent.reply(history, "Как меня зовут?", settings)
        messages = provider.calls[0][0]
        self.assertEqual(messages[0], {"role": "system", "content": "Ты технический специалист"})
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Как меня зовут?")

    def test_failed_message_is_not_returned_to_context(self):
        provider = FakeProvider()
        history = [{"role": "user", "content": "Не доставлено", "technical": {"request_status": "failed"}}]
        Agent(provider).reply(history, "Новый запрос", AgentSettings())
        self.assertEqual(len(provider.calls[0][0]), 2)

    def test_deepseek_provider_collects_full_metadata(self):
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            prompt_cache_hit_tokens=9,
            prompt_cache_miss_tokens=3,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
        message = SimpleNamespace(content="Ответ", reasoning_content="Мысли")
        response = SimpleNamespace(
            id="abc", model="deepseek-v4-pro", system_fingerprint="fp-test",
            usage=usage, choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
        result = DeepSeekProvider(client).complete([], AgentSettings(model="deepseek-v4-pro", reasoning_enabled=True))
        self.assertEqual(result.reasoning_content, "Мысли")
        self.assertEqual(result.technical["usage"]["cached_input_tokens"], 9)
        self.assertIn("elapsed_seconds", result.technical)

    def test_settings_reject_unsupported_seed_and_model_is_whitelisted(self):
        settings = AgentSettings.from_dict(AgentSettings().to_dict() | {"seed": 42})
        self.assertFalse(hasattr(settings, "seed"))
        with self.assertRaises(ValueError):
            AgentSettings.from_dict(AgentSettings().to_dict() | {"model": "unknown"})

    def test_dialogue_totals_sum_only_assistant_usage(self):
        messages = [
            {"role": "user", "technical": {"usage": {"total_tokens": 999}}},
            {"role": "assistant", "technical": {"request_status": "completed", "usage": {
                "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                "cached_input_tokens": 7, "uncached_input_tokens": 3, "reasoning_tokens": 4,
            }}},
            {"role": "assistant", "technical": {"request_status": "failed", "usage": {"total_tokens": 500}}},
        ]
        totals = dialogue_token_totals(messages)
        self.assertEqual(totals["request_count"], 1)
        self.assertEqual(totals["total_tokens"], 30)
        self.assertEqual(totals["cached_input_tokens"], 7)

    def test_summary_is_injected_into_system_prompt(self):
        provider = FakeProvider()
        Agent(provider).reply([], "Продолжай", AgentSettings(), summary="Пользователя зовут Павел.")
        sent = provider.calls[0][0]
        self.assertIn("СЖАТАЯ ПАМЯТЬ", sent[0]["content"])
        self.assertIn("Пользователя зовут Павел", sent[0]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "Продолжай"})


class StorageTests(unittest.TestCase):
    def test_conversation_survives_new_storage_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = JsonStorage(root)
            conversation = first.create_conversation("Проверка перезапуска")
            conversation["messages"].append({"role": "user", "content": "Запомни 17"})
            first.save_conversation(conversation)
            restored = JsonStorage(root).get_conversation(conversation["id"])
            self.assertEqual(restored["messages"][0]["content"], "Запомни 17")
            self.assertEqual(restored["context_management"]["mode"], "full")
            self.assertEqual(restored["summaries"], [])

    def test_preset_keeps_valid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = JsonStorage(Path(directory))
            preset = PresetManager(storage).create({
                "name": "Технический специалист",
                "description": "Точный ответ",
                "settings": AgentSettings(temperature=0.1).to_dict(),
            })
            self.assertEqual(preset["settings"]["temperature"], 0.1)
            self.assertEqual(JsonStorage(Path(directory)).load_presets_document()["presets"][0]["name"], "Технический специалист")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = FakeProvider()
        self.voice = FakeVoiceService()
        self.app = create_app(Path(self.temp.name), Agent(self.provider), self.voice)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def create_conversation(self):
        return self.client.post("/api/conversations", json={"title": "Новый диалог"}).get_json()["conversation"]

    def test_index_is_chat_not_temperature_lab(self):
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn("DeepSeek Agent", text)
        self.assertIn("Новый диалог", text)
        self.assertIn("Нет — выключен", text)
        self.assertIn("Да — включён", text)
        self.assertIn("Запрос отправлен · ожидаем DeepSeek", text)
        self.assertIn("Надиктовать сообщение", text)
        self.assertNotIn("Запустить 3 температуры", text)

    def test_local_voice_status_and_transcription(self):
        status = self.client.get("/api/voice/status").get_json()["voice"]
        self.assertTrue(status["ready"])
        started = self.client.post("/api/voice/start")
        self.assertEqual(started.status_code, 202)
        self.assertTrue(self.voice.started)
        response = self.client.post(
            "/api/voice/transcribe",
            data={"audio": (BytesIO(b"audio-bytes"), "recording.webm")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "Распознанный текст")
        self.assertEqual(self.voice.received[0], b"audio-bytes")

    def test_message_saves_reasoning_time_cache_and_settings(self):
        conversation = self.create_conversation()
        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Тест", "settings": AgentSettings(reasoning_enabled=True).to_dict()},
        )
        self.assertEqual(response.status_code, 200)
        messages = response.get_json()["conversation"]["messages"]
        assistant = messages[-1]
        self.assertEqual(assistant["reasoning_content"], "Скрытое рассуждение")
        self.assertEqual(assistant["technical"]["elapsed_seconds"], 0.321)
        self.assertEqual(assistant["technical"]["usage"]["cached_input_tokens"], 7)
        self.assertTrue(assistant["technical"]["settings"]["reasoning_enabled"])
        user = messages[-2]
        self.assertEqual(user["technical"]["token_usage"]["context_tokens"], 10)
        self.assertEqual(response.get_json()["conversation"]["token_totals"]["total_tokens"], 30)

    def test_second_request_receives_previous_dialogue(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        body = {"content": "Первый", "settings": AgentSettings().to_dict()}
        self.client.post(endpoint, json=body)
        body["content"] = "Второй"
        self.client.post(endpoint, json=body)
        sent = self.provider.calls[-1][0]
        self.assertEqual([item["content"] for item in sent[-3:]], ["Первый", "Тестовый ответ", "Второй"])
        result = self.client.get(f"/api/conversations/{conversation['id']}").get_json()["conversation"]
        self.assertEqual(result["token_totals"]["request_count"], 2)
        self.assertEqual(result["token_totals"]["input_tokens"], 20)
        self.assertEqual(result["token_totals"]["output_tokens"], 40)
        self.assertEqual(result["token_totals"]["total_tokens"], 60)

    def test_summary_mode_is_per_conversation_and_rolls_every_five_old_exchanges(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(f"{endpoint}/context", json={"mode": "summary"})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.get_json()["conversation"]["context_management"]["mode"], "summary")

        for index in range(1, 12):
            response = self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Вопрос {index}", "settings": AgentSettings().to_dict()},
            )
            self.assertEqual(response.status_code, 200)

        result = response.get_json()["conversation"]
        self.assertEqual(len(result["summaries"]), 1)
        summary = result["summaries"][0]
        self.assertEqual(summary["covered_exchange_count"], 5)
        self.assertEqual(len(summary["source_exchange_ids"]), 5)
        self.assertEqual(len(summary["source_message_ids"]), 10)
        self.assertEqual(summary["technical"]["settings"]["model"], "deepseek-v4-flash")
        self.assertFalse(summary["technical"]["settings"]["reasoning_enabled"])
        self.assertEqual(result["summary_token_totals"]["total_tokens"], 30)

        main_call_messages, _ = self.provider.calls[-1]
        sent_text = "\n".join(item["content"] for item in main_call_messages)
        sent_items = [item["content"] for item in main_call_messages]
        self.assertIn("СЖАТАЯ ПАМЯТЬ", main_call_messages[0]["content"])
        self.assertNotIn("Вопрос 1", sent_items)
        self.assertIn("Вопрос 6", sent_text)
        self.assertIn("Вопрос 11", sent_text)
        self.assertEqual(result["messages"][-1]["technical"]["context_management"]["summary_exchange_count"], 5)
        self.assertEqual(result["messages"][-1]["technical"]["context_management"]["verbatim_exchange_count"], 5)

        exported = json.loads(self.client.get(f"{endpoint}/export").data)
        self.assertEqual(len(exported["conversation"]["summaries"]), 1)
        imported = self.client.post("/api/import", json=exported).get_json()["conversation"]
        self.assertEqual(len(imported["summaries"]), 1)
        self.assertEqual(imported["context_management"]["active_summary_id"], imported["summaries"][0]["id"])

        other = self.create_conversation()
        self.assertEqual(ensure_context_management(other)["mode"], "full")

    def test_switching_back_to_full_keeps_summary_history(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.patch(f"{endpoint}/context", json={"mode": "summary"})
        for index in range(11):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Сообщение {index}", "settings": AgentSettings().to_dict()},
            )
        changed = self.client.patch(f"{endpoint}/context", json={"mode": "full"}).get_json()["conversation"]
        self.assertEqual(changed["context_management"]["mode"], "full")
        self.assertEqual(len(changed["summaries"]), 1)
        self.assertEqual(summary_token_totals(changed["summaries"])["request_count"], 1)
        self.assertEqual(len(completed_exchanges(changed["messages"])), 11)

    def test_sliding_window_sends_only_last_n_complete_exchanges(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        changed = self.client.patch(
            f"{endpoint}/context",
            json={"mode": "sliding", "sliding_window_exchanges": 2},
        )
        self.assertEqual(changed.status_code, 200)
        for index in range(1, 5):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": f"Окно {index}", "settings": AgentSettings().to_dict()},
            )
        sent = [item["content"] for item in self.provider.calls[-1][0]]
        self.assertNotIn("Окно 1", sent)
        self.assertNotIn("Окно 2", sent)
        self.assertIn("Окно 3", sent)
        self.assertEqual(sent[-1], "Окно 4")
        result = self.client.get(endpoint).get_json()["conversation"]
        self.assertEqual(result["context_management"]["sliding_window_exchanges"], 2)
        self.assertEqual(result["visible_messages"][-1]["technical"]["context_management"]["window_exchanges"], 2)

    def test_sticky_facts_auto_manual_lock_edit_and_delete(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        self.client.patch(f"{endpoint}/context", json={"mode": "facts", "facts_window_exchanges": 3})
        response = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Отвечай по-русски", "settings": AgentSettings().to_dict()},
        )
        result = response.get_json()["conversation"]
        self.assertEqual(result["facts"][0]["key"], "preferences.language")
        self.assertIn("STICKY FACTS", self.provider.calls[-1][0][0]["content"])
        self.assertEqual(result["facts_token_totals"]["request_count"], 1)

        added = self.client.post(
            f"{endpoint}/facts",
            json={"key": "goal.primary", "value": "Собрать ТЗ"},
        ).get_json()["conversation"]
        manual = next(item for item in added["facts"] if item["key"] == "goal.primary")
        self.assertTrue(manual["locked"])
        edited = self.client.put(
            f"{endpoint}/facts/{manual['id']}",
            json={"key": "goal.primary", "value": "Подготовить ТЗ", "locked": True},
        ).get_json()["conversation"]
        self.assertEqual(next(item for item in edited["facts"] if item["id"] == manual["id"])["value"], "Подготовить ТЗ")
        removed = self.client.delete(f"{endpoint}/facts/{manual['id']}")
        self.assertEqual(removed.status_code, 200)
        self.assertFalse(any(item["id"] == manual["id"] for item in removed.get_json()["conversation"]["facts"]))

    def test_branching_preserves_original_path_and_switches_numbered_siblings(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        for text in ("Первый вопрос", "Исходное продолжение"):
            self.client.post(
                f"{endpoint}/messages",
                json={"content": text, "settings": AgentSettings().to_dict()},
            )
        state = self.client.patch(f"{endpoint}/context", json={"mode": "branching"}).get_json()["conversation"]
        first_assistant = next(item for item in state["visible_messages"] if item["role"] == "assistant")
        started = self.client.post(
            f"{endpoint}/branches", json={"checkpoint_id": first_assistant["id"]},
        )
        self.assertEqual(started.status_code, 201)
        alternative = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Альтернативное продолжение", "settings": AgentSettings().to_dict()},
        ).get_json()["conversation"]
        visible_text = [item["content"] for item in alternative["visible_messages"]]
        self.assertIn("Альтернативное продолжение", visible_text)
        self.assertNotIn("Исходное продолжение", visible_text)
        point = next(item for item in alternative["branch_points"] if item["checkpoint_id"] == first_assistant["id"])
        self.assertEqual([item["number"] for item in point["options"]], [1, 2])

        original_child = point["options"][0]["child_id"]
        switched = self.client.patch(
            f"{endpoint}/branches/active",
            json={"checkpoint_id": first_assistant["id"], "child_id": original_child},
        ).get_json()["conversation"]
        switched_text = [item["content"] for item in switched["visible_messages"]]
        self.assertIn("Исходное продолжение", switched_text)
        self.assertNotIn("Альтернативное продолжение", switched_text)

    def test_branch_after_user_message_generates_new_assistant_sibling(self):
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}"
        sent = self.client.post(
            f"{endpoint}/messages",
            json={"content": "Дай вариант", "settings": AgentSettings().to_dict()},
        ).get_json()["conversation"]
        user = next(item for item in sent["visible_messages"] if item["role"] == "user")
        self.client.patch(f"{endpoint}/context", json={"mode": "branching"})
        forked = self.client.post(f"{endpoint}/branches", json={"checkpoint_id": user["id"]})
        self.assertEqual(forked.status_code, 201)
        result = forked.get_json()["conversation"]
        point = next(item for item in result["branch_points"] if item["checkpoint_id"] == user["id"])
        self.assertEqual(len(point["options"]), 2)

    def test_preset_change_creates_history_event_and_snapshot(self):
        preset_response = self.client.post("/api/presets", json={
            "name": "Специалист", "description": "", "settings": AgentSettings(temperature=0.2).to_dict(),
        })
        preset = preset_response.get_json()["preset"]
        conversation = self.create_conversation()
        endpoint = f"/api/conversations/{conversation['id']}/messages"
        self.client.post(endpoint, json={"content": "Один", "preset_id": preset["id"], "settings": preset["settings"]})
        custom = AgentSettings(temperature=1.4).to_dict()
        response = self.client.post(endpoint, json={"content": "Два", "settings": custom})
        messages = response.get_json()["conversation"]["messages"]
        self.assertTrue(any(item["role"] == "event" for item in messages))
        self.assertEqual(messages[-2]["technical"]["settings"]["temperature"], 1.4)
        self.assertEqual(messages[-2]["technical"]["configuration_source"]["type"], "custom")

    def test_export_and_import_copy_dialogue_and_used_preset(self):
        preset = self.client.post("/api/presets", json={
            "name": "Переносимый", "description": "", "settings": AgentSettings().to_dict(),
        }).get_json()["preset"]
        conversation = self.create_conversation()
        self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Экспорт", "preset_id": preset["id"], "settings": preset["settings"]},
        )
        exported = json.loads(self.client.get(f"/api/conversations/{conversation['id']}/export").data)
        imported = self.client.post("/api/import", json=exported)
        self.assertEqual(imported.status_code, 201)
        self.assertNotEqual(imported.get_json()["conversation"]["id"], conversation["id"])
        self.assertEqual(len(imported.get_json()["conversation"]["messages"]), 2)

    def test_api_failure_is_saved_but_not_fed_back(self):
        self.provider.fail = True
        conversation = self.create_conversation()
        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"content": "Ошибка", "settings": AgentSettings().to_dict()},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["conversation"]["messages"][-1]["technical"]["request_status"], "failed")
        self.assertIn("соединиться", friendly_api_error(RuntimeError("Connection error")))

    def test_context_overflow_has_clear_message(self):
        message = friendly_api_error(RuntimeError("maximum context length exceeded: too many tokens"))
        self.assertIn("превысил лимит модели", message)
        self.assertIn("сократите историю", message)

    def test_launcher_skips_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            port = occupied.getsockname()[1]
            self.assertEqual(find_available_port(port, port + 1), port + 1)


if __name__ == "__main__":
    unittest.main()
