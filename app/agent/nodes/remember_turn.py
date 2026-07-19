"""Persist a compact, structured memory after a successful query turn."""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config


def remember_turn(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    turn = int(state.get("conversation_turn") or 0) + 1
    completed_at = datetime.now(timezone.utc).isoformat()
    result_summary = state.get("result_summary") or {
        "row_count": 0,
        "columns": [],
        "preview": [],
        "truncated": False,
    }
    record = {
        "turn": turn,
        "raw_query": state.get("raw_query") or state.get("query", ""),
        "resolved_query": state.get("query", ""),
        "intent": state.get("query_intent") or {},
        "sql": state.get("sql", ""),
        "result_summary": result_summary,
        "answer": (state.get("answer_result") or {}).get("answer", ""),
        "completed_at": completed_at,
    }
    max_turns = max(1, app_config.conversation.max_history_turns)
    history = [*(state.get("conversation_history") or []), record][-max_turns:]

    runtime.stream_writer({
        "type": "conversation_memory_saved",
        "turn": turn,
        "history_size": len(history),
        "row_count": result_summary.get("row_count", 0),
        "columns": result_summary.get("columns", []),
        "answer_status": (state.get("answer_result") or {}).get("status"),
    })
    return {
        "conversation_turn": turn,
        "conversation_history": history,
        "last_query": state.get("query", ""),
        "last_query_intent": state.get("query_intent") or {},
        "last_sql": state.get("sql", ""),
        "last_result_summary": result_summary,
        "last_answer": (state.get("answer_result") or {}).get("answer", ""),
        "last_completed_at": completed_at,
    }
