"""Generate a natural-language answer and verify it against SQL evidence."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.answer_grounding import (
    build_provenance,
    deterministic_answer,
    sanitize_answer_rows,
    system_caveats,
    validate_numeric_grounding,
)
from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


def _fallback_result(
    *,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    reason: str,
    verification: dict[str, Any] | None = None,
    status: str = "fallback",
) -> dict[str, Any]:
    cfg = app_config.answer_generation
    check = verification or {
        "passed": True,
        "answer_numbers": [],
        "allowed_numbers": [],
        "invalid_numbers": [],
    }
    check = {
        **check,
        "passed": True,
        "model_output_passed": verification.get("passed")
        if verification is not None
        else None,
    }
    return {
        "status": status,
        "answer": deterministic_answer(
            rows,
            row_count=provenance["row_count"],
            max_rows=cfg.fallback_max_rows,
        ),
        "highlights": [],
        "caveats": system_caveats(provenance),
        "fallback_reason": reason,
        "verification": check,
        "provenance": provenance,
    }


def _normalize_string_list(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value[:max_items] if str(item).strip()]


async def generate_answer(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "生成可追溯回答", "status": "running"})
    cfg = app_config.answer_generation
    rows = sanitize_answer_rows(
        state.get("answer_rows") or (state.get("result_summary") or {}).get("preview") or [],
        max_rows=max(1, cfg.max_rows),
        max_cell_chars=cfg.max_cell_chars,
    )
    provenance = build_provenance(state, len(rows))

    if provenance["row_count"] == 0:
        result = _fallback_result(
            rows=rows,
            provenance=provenance,
            reason="empty_result",
            status="empty",
        )
    elif not cfg.enabled:
        result = _fallback_result(
            rows=rows,
            provenance=provenance,
            reason="answer_generation_disabled",
            status="disabled",
        )
    else:
        try:
            allowed_check = validate_numeric_grounding(
                texts=[],
                rows=rows,
                query=state.get("query", ""),
                row_count=provenance["row_count"],
                tolerance=cfg.numeric_tolerance,
            )
            prompt = PromptTemplate(
                template=load_prompt("generate_answer"),
                input_variables=[
                    "query",
                    "intent",
                    "metric_definitions",
                    "rows",
                    "row_count",
                    "truncated",
                    "allowed_numbers",
                ],
            )
            payload = await asyncio.wait_for(
                (prompt | llm | JsonOutputParser()).ainvoke({
                    "query": state.get("query", ""),
                    "intent": json.dumps(
                        state.get("query_intent") or {}, ensure_ascii=False, default=str
                    ),
                    "metric_definitions": json.dumps(
                        provenance["metrics"], ensure_ascii=False, default=str
                    ),
                    "rows": json.dumps(rows, ensure_ascii=False, default=str),
                    "row_count": provenance["row_count"],
                    "truncated": provenance["truncated"],
                    "allowed_numbers": json.dumps(
                        allowed_check["allowed_numbers"], ensure_ascii=False
                    ),
                }),
                timeout=max(0.1, cfg.timeout_seconds),
            )
            if not isinstance(payload, dict):
                raise ValueError("answer payload must be a JSON object")
            answer = str(payload.get("answer") or "").strip()
            highlights = _normalize_string_list(
                payload.get("highlights"), cfg.max_highlights
            )
            caveats = _normalize_string_list(
                payload.get("caveats"), cfg.max_highlights
            )
            if not answer:
                raise ValueError("answer is empty")
            if len(answer) > cfg.max_answer_chars:
                raise ValueError("answer exceeds configured length")
            verification = validate_numeric_grounding(
                texts=[answer, *highlights, *caveats],
                rows=rows,
                query=state.get("query", ""),
                row_count=provenance["row_count"],
                tolerance=cfg.numeric_tolerance,
            )
            if not verification["passed"]:
                result = _fallback_result(
                    rows=rows,
                    provenance=provenance,
                    reason="ungrounded_numeric_claim",
                    verification=verification,
                )
                logger.warning(
                    "回答数字校验失败，已回退确定性回答: "
                    f"invalid_numbers={verification['invalid_numbers']}"
                )
            else:
                result = {
                    "status": "generated",
                    "answer": answer,
                    "highlights": highlights,
                    "caveats": [*caveats, *system_caveats(provenance)],
                    "fallback_reason": "",
                    "verification": {
                        **verification,
                        "model_output_passed": True,
                    },
                    "provenance": provenance,
                }
        except Exception as exc:
            result = _fallback_result(
                rows=rows,
                provenance=provenance,
                reason=type(exc).__name__,
            )
            logger.warning(f"自然语言回答生成失败，已安全回退: {type(exc).__name__}")

    writer({"type": "grounded_answer", **result})
    writer({"type": "progress", "step": "生成可追溯回答", "status": "success"})
    return {"answer_result": result}
