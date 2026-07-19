"""Pause the graph for one human clarification and merge the resumed answer."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.context import DataAgentContext
from app.agent.query_intent import merge_clarification_answer
from app.agent.state import DataAgentState
from app.core.log import logger


def _answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        answer = answer.get("answer") or answer.get("value") or ""
    return str(answer or "").strip()


def clarify_intent(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    ambiguity = state.get("ambiguity_result") or {}
    question = str(ambiguity.get("question") or "请补充查询条件。")
    rounds = int(state.get("clarification_rounds") or 0)

    answer = interrupt({
        "type": "clarification_required",
        "code": ambiguity.get("code"),
        "question": question,
        "asked_slot": ambiguity.get("asked_slot"),
        "missing_slots": ambiguity.get("missing_slots") or [],
        "round": rounds + 1,
    })
    answer_text = _answer_text(answer)
    cancelled = answer_text.casefold() in {
        "取消", "算了", "停止", "cancel", "stop",
    }

    if cancelled:
        runtime.stream_writer({
            "type": "clarification_cancelled",
            "answer": answer_text,
            "round": rounds + 1,
        })
        return {
            "clarification_cancelled": True,
            "clarification_required": False,
            "clarification_answer": answer_text,
            "clarification_rounds": rounds + 1,
            "ambiguity_result": {**ambiguity, "action": "stop"},
        }

    merged_query = merge_clarification_answer(
        state.get("query", ""),
        asked_slot=ambiguity.get("asked_slot"),
        answer=answer_text,
    )
    history = [
        *(state.get("clarification_history") or []),
        {
            "round": rounds + 1,
            "asked_slot": ambiguity.get("asked_slot"),
            "question": question,
            "answer": answer_text,
            "query_before": state.get("query", ""),
            "query_after": merged_query,
        },
    ]
    runtime.stream_writer({
        "type": "clarification_resumed",
        "round": rounds + 1,
        "asked_slot": ambiguity.get("asked_slot"),
        "answer": answer_text,
        "merged_query": merged_query,
    })
    logger.info(
        f"澄清答案已合并: round={rounds + 1} "
        f"slot={ambiguity.get('asked_slot')} query={merged_query}"
    )
    return {
        "query": merged_query,
        "clarification_answer": answer_text,
        "clarification_history": history,
        "clarification_rounds": rounds + 1,
        "clarification_required": False,
        "clarification_question": "",
        "clarification_cancelled": False,
    }
