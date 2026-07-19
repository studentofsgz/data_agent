"""Score retrieval evidence before allowing SQL generation."""

from __future__ import annotations

from langgraph.runtime import Runtime

from app.agent.confidence import ConfidencePolicy, evaluate_confidence
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger


def confidence_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "评估查询置信度", "status": "running"})
    cfg = app_config.confidence

    if cfg.enabled:
        result = evaluate_confidence(
            query_intent=state.get("query_intent") or {},
            table_infos=state.get("table_infos") or [],
            metric_infos=state.get("metric_infos") or [],
            metric_semantics=state.get("metric_semantics") or {},
            column_recall_sources=state.get("column_recall_sources") or {},
            metric_recall_sources=state.get("metric_recall_sources") or {},
            column_candidate_scores=state.get("column_candidate_scores") or {},
            metric_candidate_scores=state.get("metric_candidate_scores") or {},
            schema_linking_degraded=bool(state.get("schema_linking_degraded")),
            policy=ConfidencePolicy(
                high_threshold=cfg.high_threshold,
                low_threshold=cfg.low_threshold,
                strong_similarity_score=cfg.strong_similarity_score,
                candidate_margin=cfg.candidate_margin,
            ),
        )
    else:
        result = {
            "score": 1.0,
            "level": "high",
            "action": "proceed",
            "code": "CONFIDENCE_DISABLED",
            "reasons": ["置信度门禁未启用"],
            "evidence": {},
            "interpretation": {},
            "question": "",
        }

    writer({"type": "confidence_assessment", **result})
    if result["action"] == "reject":
        writer({
            "type": "confidence_rejected",
            "code": result["code"],
            "score": result["score"],
            "message": "；".join(result["reasons"]) or "查询证据不足",
            "interpretation": result["interpretation"],
        })
        writer({"type": "progress", "step": "评估查询置信度", "status": "error"})
        logger.warning(
            f"低置信度拒绝SQL生成: code={result['code']} score={result['score']}"
        )
    elif result["action"] == "confirm":
        writer({
            "type": "confidence_confirmation_required",
            "code": result["code"],
            "score": result["score"],
            "question": result["question"],
            "interpretation": result["interpretation"],
        })
        writer({"type": "progress", "step": "评估查询置信度", "status": "waiting"})
    else:
        writer({"type": "progress", "step": "评估查询置信度", "status": "success"})

    return {
        "confidence_result": result,
        "confidence_confirmed": result["action"] == "proceed",
        "confidence_confirmation_answer": "",
    }
