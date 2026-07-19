"""Pause a medium-confidence workflow until the user confirms interpretation."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config


YES_ANSWERS = {"是", "对", "确认", "继续", "可以", "yes", "y", "ok"}
NO_ANSWERS = {"否", "不对", "取消", "停止", "不要", "no", "n", "cancel", "stop"}


def _decision(answer: Any) -> tuple[bool | None, str]:
    if isinstance(answer, dict):
        if isinstance(answer.get("confirmed"), bool):
            return answer["confirmed"], str(answer["confirmed"])
        answer = answer.get("answer") or answer.get("value") or ""
    if isinstance(answer, bool):
        return answer, str(answer)
    text = str(answer or "").strip()
    normalized = text.casefold().rstrip("。！!？?")
    if normalized in YES_ANSWERS:
        return True, text
    if normalized in NO_ANSWERS:
        return False, text
    return None, text


def confirm_confidence(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    result = state.get("confidence_result") or {}
    max_attempts = max(1, app_config.confidence.max_confirmation_attempts)

    for attempt in range(1, max_attempts + 1):
        answer = interrupt({
            "type": "confidence_confirmation_required",
            "code": result.get("code"),
            "score": result.get("score"),
            "question": result.get("question"),
            "interpretation": result.get("interpretation") or {},
            "attempt": attempt,
        })
        confirmed, answer_text = _decision(answer)
        if confirmed is True:
            runtime.stream_writer({
                "type": "confidence_confirmation_resumed",
                "confirmed": True,
                "answer": answer_text,
                "score": result.get("score"),
            })
            return {
                "confidence_confirmed": True,
                "confidence_confirmation_answer": answer_text,
                "confidence_result": {
                    **result,
                    "action": "proceed",
                    "code": "CONFIDENCE_CONFIRMED",
                },
            }
        if confirmed is False:
            runtime.stream_writer({
                "type": "confidence_confirmation_resumed",
                "confirmed": False,
                "answer": answer_text,
                "score": result.get("score"),
            })
            return {
                "confidence_confirmed": False,
                "confidence_confirmation_answer": answer_text,
                "confidence_result": {
                    **result,
                    "action": "reject",
                    "code": "CONFIDENCE_REJECTED_BY_USER",
                },
            }

    runtime.stream_writer({
        "type": "confidence_confirmation_invalid",
        "code": "CONFIDENCE_CONFIRMATION_INVALID",
        "max_attempts": max_attempts,
    })
    return {
        "confidence_confirmed": False,
        "confidence_confirmation_answer": "",
        "confidence_result": {
            **result,
            "action": "reject",
            "code": "CONFIDENCE_CONFIRMATION_INVALID",
        },
    }
