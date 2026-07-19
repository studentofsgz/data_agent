"""Deterministic confidence policy for the pre-SQL generation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DIMENSION_COLUMNS = {
    "region": "dim_region.region_name",
    "province": "dim_region.province",
    "city": "dim_region.city",
    "category": "dim_product.category",
    "brand": "dim_product.brand",
    "member_level": "dim_customer.member_level",
    "gender": "dim_customer.gender",
}

METRIC_REQUIRED_COLUMNS = {
    "GMV": {"fact_order.order_amount"},
    "AOV": {"fact_order.order_amount", "fact_order.order_id"},
    "ORDER_COUNT": {"fact_order.order_id"},
    "SALES_QUANTITY": {"fact_order.order_quantity"},
    "AVG_PURCHASE_QUANTITY": {"fact_order.order_quantity"},
    "ORDER_AMOUNT": {"fact_order.order_amount"},
}


@dataclass(frozen=True)
class ConfidencePolicy:
    high_threshold: float = 0.75
    low_threshold: float = 0.45
    strong_similarity_score: float = 0.55
    candidate_margin: float = 0.05


def _final_column_ids(table_infos: list[dict[str, Any]]) -> set[str]:
    return {
        f"{table.get('name')}.{column.get('name')}"
        for table in table_infos
        for column in table.get("columns") or []
        if table.get("name") and column.get("name")
    }


def _max_score(
    scores: dict[str, float | None],
    allowed_ids: set[str],
) -> float | None:
    values = [
        float(score)
        for item_id, score in scores.items()
        if item_id in allowed_ids and score is not None
    ]
    return max(values) if values else None


def _has_close_metric_candidates(
    scores: dict[str, float | None],
    final_metrics: set[str],
    margin: float,
) -> bool:
    values = sorted(
        (
            float(score)
            for metric_id, score in scores.items()
            if metric_id in final_metrics and score is not None
        ),
        reverse=True,
    )
    return len(values) > 1 and values[0] - values[1] < margin


def _build_interpretation(
    intent: dict[str, Any],
    table_infos: list[dict[str, Any]],
    metric_infos: list[dict[str, Any]],
    semantic_metrics: set[str],
) -> dict[str, Any]:
    time_info = intent.get("time") or {}
    return {
        "metrics": sorted(
            semantic_metrics
            or {str(item.get("name")) for item in metric_infos if item.get("name")}
            or set(intent.get("metrics") or [])
        ),
        "dimensions": list(intent.get("dimensions") or []),
        "time": {
            key: time_info.get(key)
            for key in ("year", "month", "day", "relative", "grain")
            if time_info.get(key) is not None
        },
        "filters": list(intent.get("filters") or []),
        "tables": sorted(
            str(table.get("name"))
            for table in table_infos
            if table.get("name")
        ),
    }


def _confirmation_question(interpretation: dict[str, Any]) -> str:
    metrics = "、".join(interpretation["metrics"]) or "明细字段"
    dimensions = "、".join(interpretation["dimensions"]) or "不分组"
    tables = "、".join(interpretation["tables"]) or "未确定"
    return (
        f"当前证据不足以直接执行。我理解为查询指标“{metrics}”，"
        f"分析维度“{dimensions}”，使用表“{tables}”。是否按这个理解继续？"
    )


def evaluate_confidence(
    *,
    query_intent: dict[str, Any],
    table_infos: list[dict[str, Any]],
    metric_infos: list[dict[str, Any]],
    metric_semantics: dict[str, Any],
    column_recall_sources: dict[str, list[str]],
    metric_recall_sources: dict[str, list[str]],
    column_candidate_scores: dict[str, float | None],
    metric_candidate_scores: dict[str, float | None],
    schema_linking_degraded: bool = False,
    policy: ConfidencePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ConfidencePolicy()
    final_columns = _final_column_ids(table_infos)
    final_metrics = {
        str(item.get("name"))
        for item in metric_infos
        if item.get("name")
    }
    semantic_metrics = {
        str(item.get("name"))
        for item in metric_semantics.get("metrics") or []
        if item.get("name")
    }
    intent_metrics = set(query_intent.get("metrics") or [])
    unknown_metrics = list(
        query_intent.get("unresolved_metric_mentions") or []
    )

    expected_dimensions = {
        DIMENSION_COLUMNS[dimension]
        for dimension in query_intent.get("dimensions") or []
        if dimension in DIMENSION_COLUMNS
    }
    missing_dimensions = sorted(expected_dimensions - final_columns)

    required_metric_columns = set().union(
        *(METRIC_REQUIRED_COLUMNS.get(metric, set()) for metric in intent_metrics)
    ) if intent_metrics else set()
    metric_column_coverage = (
        len(required_metric_columns & final_columns) / len(required_metric_columns)
        if required_metric_columns
        else 1.0
    )
    semantic_metric_coverage = (
        len(intent_metrics & semantic_metrics) / len(intent_metrics)
        if intent_metrics
        else 1.0
    )

    exact_columns = {
        item_id
        for item_id, sources in column_recall_sources.items()
        if item_id in final_columns and "exact_alias" in sources
    }
    exact_metrics = {
        item_id
        for item_id, sources in metric_recall_sources.items()
        if item_id in final_metrics and "exact_alias" in sources
    }
    max_column_score = _max_score(column_candidate_scores, final_columns)
    max_metric_score = _max_score(metric_candidate_scores, final_metrics)
    metric_conflict = (
        len(final_metrics) > max(1, len(intent_metrics))
        and not (intent_metrics & exact_metrics)
    ) or (
        _has_close_metric_candidates(
            metric_candidate_scores,
            final_metrics,
            policy.candidate_margin,
        )
        and not (intent_metrics & exact_metrics)
    )

    evidence = {
        "intent_metrics": sorted(intent_metrics),
        "unknown_metric_mentions": unknown_metrics,
        "semantic_metrics": sorted(semantic_metrics),
        "final_metrics": sorted(final_metrics),
        "final_tables": sorted(
            str(table.get("name"))
            for table in table_infos
            if table.get("name")
        ),
        "final_column_count": len(final_columns),
        "missing_dimensions": missing_dimensions,
        "required_metric_column_coverage": round(metric_column_coverage, 4),
        "semantic_metric_coverage": round(semantic_metric_coverage, 4),
        "exact_column_count": len(exact_columns),
        "exact_metric_count": len(exact_metrics),
        "max_column_score": max_column_score,
        "max_metric_score": max_metric_score,
        "metric_candidate_conflict": metric_conflict,
        "schema_linking_degraded": schema_linking_degraded,
    }
    interpretation = _build_interpretation(
        query_intent,
        table_infos,
        metric_infos,
        semantic_metrics,
    )

    hard_rejection: tuple[str, str] | None = None
    if unknown_metrics:
        hard_rejection = (
            "UNKNOWN_METRIC",
            "问题包含语义层未定义的业务指标：" + "、".join(unknown_metrics),
        )
    elif not table_infos or not final_columns:
        hard_rejection = (
            "NO_SCHEMA_CONTEXT",
            "没有形成可用于SQL生成的表字段上下文",
        )
    elif missing_dimensions:
        hard_rejection = (
            "MISSING_REQUIRED_DIMENSION",
            "最终Schema缺少查询要求的维度字段：" + "、".join(missing_dimensions),
        )
    elif required_metric_columns and metric_column_coverage < 1.0:
        hard_rejection = (
            "MISSING_METRIC_EVIDENCE",
            "最终Schema缺少指标计算所需字段",
        )

    score = 0.25
    score += 0.20 if table_infos and final_columns else 0.0
    score += 0.15 if not expected_dimensions or not missing_dimensions else 0.0
    if intent_metrics:
        if semantic_metric_coverage == 1.0 and semantic_metrics:
            score += 0.30
        elif metric_column_coverage == 1.0:
            score += 0.24
        elif intent_metrics & final_metrics:
            score += 0.18
    else:
        score += 0.12
    score += 0.08 if exact_columns else 0.0
    score += 0.08 if exact_metrics else 0.0
    score += (
        0.07
        if max_column_score is not None
        and max_column_score >= policy.strong_similarity_score
        else 0.0
    )
    score += (
        0.07
        if max_metric_score is not None
        and max_metric_score >= policy.strong_similarity_score
        else 0.0
    )
    if schema_linking_degraded:
        score -= 0.18
    if metric_conflict:
        score -= 0.18
    if not exact_columns and (
        max_column_score is None
        or max_column_score < policy.strong_similarity_score
    ):
        score -= 0.08
    score = round(min(1.0, max(0.0, score)), 4)

    reasons: list[str] = []
    if semantic_metrics:
        reasons.append("命中确定性指标语义层")
    if exact_columns or exact_metrics:
        reasons.append("存在精确名称或别名命中")
    if schema_linking_degraded:
        reasons.append("Schema Linking重排发生降级")
    if metric_conflict:
        reasons.append("存在多个接近的指标候选")

    if hard_rejection:
        code, reason = hard_rejection
        reasons.append(reason)
        score = min(score, max(0.0, policy.low_threshold - 0.01))
        return {
            "score": score,
            "level": "low",
            "action": "reject",
            "code": code,
            "reasons": reasons,
            "evidence": evidence,
            "interpretation": interpretation,
            "question": "",
        }

    if score >= policy.high_threshold:
        return {
            "score": score,
            "level": "high",
            "action": "proceed",
            "code": "CONFIDENCE_HIGH",
            "reasons": reasons,
            "evidence": evidence,
            "interpretation": interpretation,
            "question": "",
        }
    if score < policy.low_threshold:
        reasons.append("可验证证据低于安全执行阈值")
        return {
            "score": score,
            "level": "low",
            "action": "reject",
            "code": "CONFIDENCE_TOO_LOW",
            "reasons": reasons,
            "evidence": evidence,
            "interpretation": interpretation,
            "question": "",
        }

    return {
        "score": score,
        "level": "medium",
        "action": "confirm",
        "code": "CONFIRM_INTERPRETATION",
        "reasons": reasons,
        "evidence": evidence,
        "interpretation": interpretation,
        "question": _confirmation_question(interpretation),
    }
