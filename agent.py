"""Инкапсулированный агент и провайдер DeepSeek."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from openai import OpenAI


DEEPSEEK_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)


PROVIDER_CAPABILITIES: dict[str, Any] = {
    "provider": "deepseek",
    "models": list(DEEPSEEK_MODELS),
    "controls": {
        "system_prompt": {"type": "text", "max_length": 20_000},
        "temperature": {"type": "number", "min": 0, "max": 2, "step": 0.01},
        "top_p": {"type": "number", "min": 0, "max": 1, "step": 0.01},
        "reasoning_enabled": {"type": "boolean"},
        "reasoning_effort": {"type": "select", "options": ["low", "high", "max"]},
        "max_tokens": {"type": "optional_integer", "min": 1, "max": 65_536},
        "stop": {"type": "optional_string_list", "max_items": 16},
        "response_format": {"type": "select", "options": ["text", "json_object"]},
        "logprobs": {"type": "boolean"},
        "top_logprobs": {"type": "optional_integer", "min": 0, "max": 20},
    },
    "unsupported": ["seed", "frequency_penalty", "presence_penalty"],
}


@dataclass(frozen=True)
class AgentSettings:
    model: str = "deepseek-v4-flash"
    system_prompt: str = "Ты полезный и внимательный ассистент."
    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_enabled: bool = False
    reasoning_effort: str = "high"
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    response_format: str = "text"
    logprobs: bool = False
    top_logprobs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "AgentSettings":
        if not isinstance(value, dict):
            raise ValueError("Настройки агента должны быть JSON-объектом.")

        model = str(value.get("model", cls.model)).strip()
        if model not in DEEPSEEK_MODELS:
            raise ValueError("Выбрана неподдерживаемая модель DeepSeek.")

        system_prompt = str(value.get("system_prompt", cls.system_prompt)).strip()
        if not system_prompt:
            raise ValueError("Системный промпт не должен быть пустым.")
        if len(system_prompt) > 20_000:
            raise ValueError("Системный промпт длиннее 20 000 символов.")

        temperature = _number(value.get("temperature", 1), "Температура", 0, 2)
        top_p = _number(value.get("top_p", 1), "top_p", 0, 1)
        reasoning_enabled = _boolean(value.get("reasoning_enabled", False), "Reasoning")
        reasoning_effort = str(value.get("reasoning_effort", "high"))
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort должен быть low, high или max.")

        max_tokens = _optional_integer(value.get("max_tokens"), "Лимит токенов", 1, 65_536)
        response_format = str(value.get("response_format", "text"))
        if response_format not in {"text", "json_object"}:
            raise ValueError("Формат ответа должен быть text или json_object.")

        logprobs = _boolean(value.get("logprobs", False), "logprobs")
        top_logprobs = _optional_integer(value.get("top_logprobs"), "top_logprobs", 0, 20)
        if top_logprobs is not None and not logprobs:
            raise ValueError("top_logprobs можно задать только при включённом logprobs.")

        raw_stop = value.get("stop", [])
        if raw_stop in (None, ""):
            stop: list[str] = []
        elif isinstance(raw_stop, list):
            stop = [str(item) for item in raw_stop if str(item)]
        else:
            raise ValueError("Стоп-последовательности должны быть списком строк.")
        if len(stop) > 16 or any(len(item) > 500 for item in stop):
            raise ValueError("Разрешено до 16 стоп-последовательностей длиной до 500 символов.")

        return cls(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )


@dataclass(frozen=True)
class AgentResult:
    content: str
    reasoning_content: str
    technical: dict[str, Any]


class ChatProvider(Protocol):
    def complete(self, messages: list[dict[str, str]], settings: AgentSettings) -> AgentResult: ...


class DeepSeekProvider:
    """Тонкий адаптер DeepSeek. Его можно заменить другим провайдером."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Не найден DEEPSEEK_API_KEY. Добавьте ключ в файл .env и перезапустите приложение."
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=180.0,
            max_retries=1,
        )
        return self._client

    def complete(self, messages: list[dict[str, str]], settings: AgentSettings) -> AgentResult:
        request_args: dict[str, Any] = {
            "model": settings.model,
            "messages": messages,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "stream": False,
            "response_format": {"type": settings.response_format},
            "logprobs": settings.logprobs,
            "extra_body": {
                "thinking": {"type": "enabled" if settings.reasoning_enabled else "disabled"},
                "reasoning_effort": settings.reasoning_effort,
            },
        }
        if settings.max_tokens is not None:
            request_args["max_tokens"] = settings.max_tokens
        if settings.stop:
            request_args["stop"] = settings.stop
        if settings.top_logprobs is not None:
            request_args["top_logprobs"] = settings.top_logprobs

        started = time.perf_counter()
        response = self._get_client().chat.completions.create(**request_args)
        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        technical = {
            "provider": "deepseek",
            "request_id": getattr(response, "id", None),
            "model": getattr(response, "model", None) or settings.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "elapsed_seconds": round(elapsed, 3),
            "usage": {
                "input_tokens": _usage_value(usage, "prompt_tokens"),
                "output_tokens": _usage_value(usage, "completion_tokens"),
                "total_tokens": _usage_value(usage, "total_tokens"),
                "cached_input_tokens": _usage_value(usage, "prompt_cache_hit_tokens"),
                "uncached_input_tokens": _usage_value(usage, "prompt_cache_miss_tokens"),
                "reasoning_tokens": _usage_value(completion_details, "reasoning_tokens"),
            },
            "settings": settings.to_dict(),
        }
        return AgentResult(
            content=getattr(message, "content", None) or "(Модель не вернула текст ответа)",
            reasoning_content=getattr(message, "reasoning_content", None) or "",
            technical=technical,
        )


class Agent:
    """Самостоятельная сущность, управляющая контекстом запроса и ответом LLM."""

    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def reply(
        self,
        history: list[dict[str, Any]],
        user_text: str,
        settings: AgentSettings,
        summary: str = "",
        facts: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        system_prompt = settings.system_prompt
        if summary:
            system_prompt += (
                "\n\nСЖАТАЯ ПАМЯТЬ ПРЕДЫДУЩЕГО ДИАЛОГА:\n"
                f"{summary}\n\n"
                "Используй эту память как контекст разговора. Более новые сообщения ниже имеют приоритет."
            )
        if facts:
            fact_lines = "\n".join(f"- {item['key']} = {item['value']}" for item in facts)
            system_prompt += (
                "\n\nSTICKY FACTS — ВАЖНЫЕ ДАННЫЕ ДИАЛОГА:\n"
                f"{fact_lines}\n\n"
                "Считай эти пары ключ-значение устойчивой памятью. "
                "Более новые сообщения пользователя имеют приоритет при явном противоречии."
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in history:
            role = item.get("role")
            content = item.get("content")
            technical = item.get("technical", {})
            if technical.get("request_status") == "failed":
                continue
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return self.provider.complete(messages, settings)

    def extract_facts(self, current_facts: list[dict[str, Any]], user_text: str) -> AgentResult:
        """Возвращает обновлённую key-value память в строгом JSON."""
        facts_json = json.dumps(current_facts, ensure_ascii=False, indent=2)
        prompt = (
            "Текущие facts:\n"
            f"{facts_json}\n\n"
            "Новое сообщение пользователя:\n"
            f"{user_text}\n\n"
            "Верни JSON-объект строго вида "
            '{"facts":[{"key":"category.name","value":"точное значение"}]}. '
            "Это должен быть полный актуальный набор, а не только изменения. "
            "Разрешённые категории ключей: goal, context, entities, requirements, constraints, "
            "preferences, decisions, agreements, resources, environment, numbers, deadlines, "
            "definitions, output, open_questions, next_steps, risks. "
            "Ключи с locked=true сохрани дословно. Удаляй только явно устаревшие незакреплённые данные. "
            "Не выдумывай фактов и не добавляй пояснений вне JSON."
        )
        settings = AgentSettings(
            model="deepseek-v4-flash",
            system_prompt="Ты обновляешь точную структурированную память диалога.",
            temperature=0.1,
            top_p=1.0,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=2_000,
            response_format="json_object",
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)

    def summarize(self, previous_summary: str, exchanges: list[dict[str, Any]]) -> AgentResult:
        """Обновляет накопительную память дешёвой моделью без reasoning."""
        transcript: list[str] = []
        for exchange in exchanges:
            for message in exchange.get("messages", []):
                label = "Пользователь" if message.get("role") == "user" else "Ассистент"
                transcript.append(f"{label}: {message.get('content', '')}")
        previous = previous_summary or "(предыдущего summary ещё нет)"
        joined_transcript = "\n\n".join(transcript)
        prompt = (
            "Предыдущий накопительный summary:\n"
            f"{previous}\n\n"
            "Новые обмены для добавления:\n"
            f"{joined_transcript}\n\n"
            "Верни новый цельный summary на русском языке. Сохрани факты, имена, числа, решения, "
            "предпочтения пользователя, незавершённые задачи и важные ограничения. Не добавляй домыслов. "
            "Не описывай сам процесс суммаризации и не используй JSON."
        )
        settings = AgentSettings(
            model="deepseek-v4-flash",
            system_prompt="Ты создаёшь точную компактную память длительного диалога.",
            temperature=0.2,
            top_p=1.0,
            reasoning_enabled=False,
            reasoning_effort="low",
            max_tokens=2_000,
        )
        return self.provider.complete([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ], settings)


TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "reasoning_tokens",
)


def normalize_token_usage(value: Any) -> dict[str, int]:
    """Возвращает безопасный набор целочисленных счётчиков одного API-вызова."""
    source = value if isinstance(value, dict) else {}
    result: dict[str, int] = {}
    for field_name in TOKEN_USAGE_FIELDS:
        raw_value = source.get(field_name, 0)
        try:
            result[field_name] = max(0, int(raw_value or 0))
        except (TypeError, ValueError):
            result[field_name] = 0
    if result["total_tokens"] == 0:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def dialogue_token_totals(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Суммирует фактический usage всех успешно завершённых вызовов диалога."""
    totals = {field_name: 0 for field_name in TOKEN_USAGE_FIELDS}
    totals["request_count"] = 0
    for message in messages:
        technical = message.get("technical")
        if message.get("role") != "assistant" or not isinstance(technical, dict):
            continue
        if technical.get("request_status") == "failed" or not isinstance(technical.get("usage"), dict):
            continue
        usage = normalize_token_usage(technical["usage"])
        totals["request_count"] += 1
        for field_name in TOKEN_USAGE_FIELDS:
            totals[field_name] += usage[field_name]
    return totals


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} должна быть числом от {minimum:g} до {maximum:g}.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} должна быть от {minimum:g} до {maximum:g}.")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} должен быть логическим значением.")
    return value


def _optional_integer(value: Any, label: str, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    return value


def _usage_value(source: Any, name: str) -> int:
    value = getattr(source, name, 0) if source is not None else 0
    return int(value or 0)
