"""Политика допустимого поведения агента на этапах задачи."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from task_state import EVENT_LABELS, allowed_task_events


STAGE_ACTIONS = {
    "planning": ("clarification", "analysis", "planning", "request_plan_approval", "refusal"),
    "execution": ("clarification", "implementation", "progress_report", "request_validation", "request_rollback", "refusal"),
    "validation": ("clarification", "validation", "defect_report", "validation_result", "refusal"),
    "done": ("clarification", "final_summary", "result_explanation", "refusal"),
}

ACTION_LABELS = {
    "clarification": "уточнение",
    "analysis": "анализ",
    "planning": "подготовка плана",
    "request_plan_approval": "запрос утверждения плана",
    "implementation": "реализация",
    "progress_report": "отчёт о выполнении",
    "request_validation": "запрос проверки",
    "request_rollback": "запрос возврата",
    "validation": "проверка",
    "defect_report": "отчёт о дефектах",
    "validation_result": "результат проверки",
    "final_summary": "итог",
    "result_explanation": "объяснение результата",
    "refusal": "контролируемый отказ",
}


class PolicyValidationError(ValueError):
    """Проверяющая модель вернула результат, который нельзя безопасно принять."""


def active_invariants(invariant_bundle: dict[str, Any] | None) -> list[dict[str, str]]:
    if not invariant_bundle:
        return []
    return [deepcopy(item) for item in invariant_bundle.get("rules", []) if isinstance(item, dict) and item.get("text")]


def policy_context(task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> dict[str, Any]:
    stage = str(task_state.get("stage") or "planning")
    events = [event for event in allowed_task_events(task_state) if event not in {"pause", "resume"}]
    invariants = []
    for index, item in enumerate(active_invariants(invariant_set), start=1):
        invariants.append({**item, "ref": f"I{index}"})
    return {
        "stage": stage,
        "lifecycle_mode": task_state.get("lifecycle_mode", "controlled"),
        "allowed_action_types": list(STAGE_ACTIONS.get(stage, ("refusal",))),
        "allowed_events": events,
        "invariants": invariants,
    }


def generation_guidance(task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> str:
    context = policy_context(task_state, invariant_set)
    invariant_lines = "\n".join(
        f"- {item['ref']}: {item['text']}" for item in context["invariants"]
    ) or "- Активных инвариантов нет."
    return (
        "\n\nОГРАНИЧЕНИЯ КОНТРОЛИРУЕМОГО ОТВЕТА:\n"
        "Верни обычный ответ пользователю. Не добавляй служебный JSON, самоотчёт, action_type, "
        "checked_invariant_ids или другие поля внутреннего контроля.\n"
        f"Разрешённые action_type на текущем этапе: {', '.join(context['allowed_action_types'])}.\n"
        "Учитывай каждый активный инвариант ниже. Если запрос требует запрещённого этапом действия или нарушает "
        "инвариант, откажись выполнять запрещённую часть и понятно объясни причину. Не выполняй запрещённое действие "
        "внутри текста отказа. Переход состояния выполняет только сервер.\n"
        "Активные инварианты:\n" + invariant_lines
    )


def validation_prompt(
    user_text: str,
    draft: str,
    task_state: dict[str, Any],
    invariant_set: dict[str, Any] | None,
) -> str:
    context = policy_context(task_state, invariant_set)
    return (
        "Независимо проверь черновик ответа перед показом пользователю. Не продолжай задачу и не улучшай ответ. "
        "Определи фактический тип действия по содержанию, а не по заявленной метке. Проверь этап и каждый инвариант. "
        "Для ссылок на инварианты используй только короткие значения ref (I1, I2, ...) из политики. "
        "Верни только JSON вида "
        '{"allowed":true,"detected_action_type":"planning","checked_invariant_ids":["I1"],'
        '"violated_invariant_ids":[],"explanation":""}. '
        "checked_invariant_ids должен содержать ref каждого переданного активного инварианта. "
        "allowed=true допустимо только если фактическое действие разрешено этапом и инварианты не нарушены.\n\n"
        f"Запрос пользователя:\n{user_text}\n\n"
        f"Черновик ответа:\n{draft}\n\n"
        f"Политика:\n{json.dumps(context, ensure_ascii=False)}"
    )


def parse_validation_result(content: str, task_state: dict[str, Any], invariant_set: dict[str, Any] | None) -> dict[str, Any]:
    value = _json_object(content, "Проверяющая модель не вернула структурированный результат.")
    context = policy_context(task_state, invariant_set)
    allowed = value.get("allowed")
    action = value.get("detected_action_type")
    checked_refs = _string_list(value.get("checked_invariant_ids"))
    checked, unknown_checked = _resolve_invariant_refs(checked_refs, context)
    required_ids = [item["id"] for item in context["invariants"]]
    if set(checked) != set(required_ids):
        raise PolicyValidationError("Проверяющая модель не подтвердила проверку всех активных инвариантов.")
    violation_refs = _string_list(value.get("violated_invariant_ids"))
    violations, unknown_violations = _resolve_invariant_refs(violation_refs, context)
    explanation = str(value.get("explanation") or "").strip()[:2_000]
    if not isinstance(allowed, bool) or action not in ACTION_LABELS:
        raise PolicyValidationError("Проверяющая модель вернула неполный результат.")
    server_allowed = action in context["allowed_action_types"] and not violations and not unknown_violations
    return {
        "allowed": bool(allowed and server_allowed),
        "detected_action_type": action,
        "checked_invariant_ids": checked,
        "violated_invariant_ids": violations,
        "unknown_invariant_refs": list(dict.fromkeys(unknown_checked + unknown_violations)),
        "explanation": explanation or (
            "Проверка вернула неизвестную ссылку на инвариант." if unknown_violations else ""
        ),
    }


def blocked_content(reason: str, task_state: dict[str, Any], invariant_set: dict[str, Any] | None, violations: list[str] | None = None) -> str:
    stage = str(task_state.get("stage") or "planning")
    rules = {item["id"]: item["text"] for item in active_invariants(invariant_set)}
    lines = ["Ответ заблокирован контроллером состояния задачи.", f"Текущий этап: {stage}.", reason]
    for rule_id in violations or []:
        if rule_id in rules:
            lines.append(f"Нарушенный инвариант: {rules[rule_id]}")
    lines.append("Измените запрос либо выполните разрешённый переход состояния.")
    return "\n".join(lines)


def policy_audit(
    validation: dict[str, Any] | None,
    task_state: dict[str, Any],
    invariant_set: dict[str, Any] | None,
    *,
    accepted: bool,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "stage": task_state.get("stage"),
        "lifecycle_mode": task_state.get("lifecycle_mode"),
        "detected_action_type": (validation or {}).get("detected_action_type"),
        "checked_invariants": deepcopy(active_invariants(invariant_set)),
        "checked_invariant_ids": list((validation or {}).get("checked_invariant_ids", [])),
        "violated_invariant_ids": list((validation or {}).get("violated_invariant_ids", [])),
        "unknown_invariant_refs": list((validation or {}).get("unknown_invariant_refs", [])),
        "reason": reason,
    }


def _json_object(content: str, message: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise PolicyValidationError(message)
    text = content.strip().lstrip("\ufeff")
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            candidates.append(text[first_newline + 1:-3].strip())
    opening = text.find("{")
    closing = text.rfind("}")
    if opening >= 0 and closing > opening:
        candidates.append(text[opening:closing + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PolicyValidationError(message)


def _resolve_invariant_refs(values: list[str], context: dict[str, Any]) -> tuple[list[str], list[str]]:
    aliases: dict[str, str] = {}
    for item in context.get("invariants", []):
        invariant_id = str(item.get("id") or "")
        reference = str(item.get("ref") or "")
        if invariant_id:
            aliases[invariant_id] = invariant_id
        if reference:
            aliases[reference] = invariant_id
    resolved = [aliases[value] for value in values if value in aliases]
    unknown = [value for value in values if value not in aliases]
    return list(dict.fromkeys(resolved)), list(dict.fromkeys(unknown))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolicyValidationError("Ожидался список строк в результате проверки политики.")
    return list(dict.fromkeys(value))
