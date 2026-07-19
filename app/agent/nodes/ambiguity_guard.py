"""Emit a minimal clarification request before retrieval when intent is incomplete."""

from __future__ import annotations

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.query_intent import AmbiguityPolicy, analyze_query_intent
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger


def ambiguity_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "检查问题完整性", "status": "running"})

    cfg = app_config.ambiguity
    intent, ambiguity = analyze_query_intent(
        state.get("query", ""),
        has_history=bool(state.get("messages")),
        policy=AmbiguityPolicy(
            require_year_for_explicit_month=cfg.require_year_for_explicit_month,
            clarify_vague_metric=cfg.clarify_vague_metric,
            clarify_vague_time=cfg.clarify_vague_time,
            clarify_vague_top_k=cfg.clarify_vague_top_k,
        ),
    )
    action = (
        "clarify"
        if cfg.enabled and cfg.stop_on_ambiguity and ambiguity["needs_clarification"]
        else "continue"
    )
    result = {**ambiguity, "action": action}

    writer({
        "type": "query_intent",
        "intent": intent,
        "ambiguity": result,
    })

    if action == "clarify":
        writer({
            "type": "clarification_required",
            "code": result["code"],
            "question": result["question"],
            "missing_slots": result["missing_slots"],
            "reasons": result["reasons"],
            "intent": intent,
        })
        writer({"type": "progress", "step": "检查问题完整性", "status": "waiting"})
        logger.info(
            f"问题需要澄清: code={result['code']} "
            f"missing_slots={result['missing_slots']}"
        )
    else:
        writer({"type": "progress", "step": "检查问题完整性", "status": "success"})

    return {
        "query_intent": intent,
        "ambiguity_result": result,
        "clarification_required": action == "clarify",
        "clarification_question": result["question"] if action == "clarify" else "",
    }
