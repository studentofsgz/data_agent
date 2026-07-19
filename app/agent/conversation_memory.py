"""Deterministic short-term conversation memory for follow-up analytics queries."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.agent.query_intent import extract_query_intent


FOLLOW_UP_PATTERN = re.compile(
    r"(?:^(?:那|那么|其中|再|只|改成|换成|继续)|"
    r"(?:呢|怎么样)[？?]?$|这个指标|那个指标|上述|刚才)"
)

METRIC_LABELS = {
    "GMV": "销售额",
    "AOV": "客单价",
    "ORDER_COUNT": "订单量",
    "SALES_QUANTITY": "销量",
    "AVG_PURCHASE_QUANTITY": "平均购买数量",
    "ORDER_AMOUNT": "订单金额",
}

DIMENSION_LABELS = {
    "region": "地区",
    "province": "省份",
    "city": "城市",
    "product": "商品",
    "category": "品类",
    "brand": "品牌",
    "customer": "客户",
    "member_level": "会员等级",
    "gender": "性别",
}

RELATIVE_TIME_LABELS = {
    "last_month": "上个月",
    "this_month": "本月",
    "next_month": "下个月",
    "last_year": "去年",
    "this_year": "今年",
    "next_year": "明年",
    "yesterday": "昨天",
    "today": "今天",
}


def is_context_dependent(query: str) -> bool:
    return bool(FOLLOW_UP_PATTERN.search(str(query or "").strip()))


def _explicit_grain(query: str) -> str | None:
    text = str(query or "")
    if re.search(r"每天|每日|按日|逐日", text):
        return "day"
    if re.search(r"每个月|每月|按月|月度|逐月", text):
        return "month"
    if re.search(r"每季度|按季度|季度", text):
        return "quarter"
    if re.search(r"每年|按年|年度|逐年", text):
        return "year"
    return None


def _has_recognized_change(intent: dict[str, Any]) -> bool:
    time_info = intent.get("time") or {}
    return bool(
        intent.get("metrics")
        or intent.get("dimensions")
        or intent.get("filters")
        or intent.get("top_k") is not None
        or intent.get("order")
        or any(time_info.get(key) is not None for key in ("year", "month", "day", "relative", "grain"))
    )


def _merge_time(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    inherited: list[str] = []
    overridden: list[str] = []

    if current.get("relative"):
        merged = {
            "year": None,
            "month": None,
            "day": None,
            "relative": current["relative"],
            "grain": current.get("grain") or previous.get("grain"),
        }
        overridden.append("time.relative")
        if current.get("grain") is None and previous.get("grain"):
            inherited.append("time.grain")
        return merged, inherited, overridden

    merged: dict[str, Any] = {"relative": None}
    for key in ("year", "month", "day", "grain"):
        if current.get(key) is not None:
            merged[key] = current[key]
            overridden.append(f"time.{key}")
        else:
            merged[key] = previous.get(key)
            if previous.get(key) is not None:
                inherited.append(f"time.{key}")

    if not any(current.get(key) is not None for key in ("year", "month", "day", "grain")):
        merged["relative"] = previous.get("relative")
        if previous.get("relative"):
            inherited.append("time.relative")
    return merged, inherited, overridden


def _merge_filters(
    previous: list[dict[str, str]],
    current: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    merged = {item["field"]: dict(item) for item in previous}
    inherited = [f"filter.{field}" for field in merged]
    overridden: list[str] = []
    for item in current:
        field = item["field"]
        merged[field] = dict(item)
        overridden.append(f"filter.{field}")
        inherited = [slot for slot in inherited if slot != f"filter.{field}"]
    return list(merged.values()), inherited, overridden


def _render_time(time_info: dict[str, Any]) -> str:
    relative = time_info.get("relative")
    if relative:
        if relative.startswith("recent_"):
            _, amount, unit = relative.split("_", 2)
            return f"最近{amount}{unit}"
        return RELATIVE_TIME_LABELS.get(relative, relative)

    parts: list[str] = []
    if time_info.get("year") is not None:
        parts.append(f"{time_info['year']}年")
    if time_info.get("month") is not None:
        parts.append(f"{time_info['month']}月")
    if time_info.get("day") is not None:
        parts.append(f"{time_info['day']}日")
    return "".join(parts)


def _render_filter(item: dict[str, str]) -> str:
    field = item.get("field")
    value = str(item.get("value") or "")
    if field == "gender":
        return {"female": "女性客户", "male": "男性客户"}.get(value, value)
    if field == "member_level":
        return value
    if field == "region":
        return f"{value}地区"
    return value


def render_query_intent(intent: dict[str, Any]) -> str:
    parts: list[str] = []
    time_text = _render_time(intent.get("time") or {})
    if time_text:
        parts.append(time_text)

    filters = [_render_filter(item) for item in intent.get("filters") or []]
    parts.extend(item for item in filters if item)

    dimensions = [
        DIMENSION_LABELS.get(item, item)
        for item in intent.get("dimensions") or []
    ]
    if dimensions:
        parts.append(f"按{'、'.join(dimensions)}")

    metrics = [
        METRIC_LABELS.get(item, item)
        for item in intent.get("metrics") or []
    ]
    if metrics:
        parts.append(f"统计{'和'.join(metrics)}")

    grain = (intent.get("time") or {}).get("grain")
    grain_text = {
        "day": "按天展示",
        "month": "按月展示",
        "quarter": "按季度展示",
        "year": "按年展示",
    }.get(grain)
    if grain_text:
        parts.append(grain_text)

    top_k = intent.get("top_k")
    order = intent.get("order")
    if top_k is not None:
        parts.append(f"按指标{'升序' if order == 'ASC' else '降序'}取前{top_k}名")
    elif order:
        parts.append(f"按指标{'升序' if order == 'ASC' else '降序'}")

    return "，".join(parts)


def resolve_structured_followup(
    query: str,
    previous_intent: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve supported follow-ups without an LLM; return an audit-friendly decision."""
    raw_query = str(query or "").strip()
    if not previous_intent or not is_context_dependent(raw_query):
        return {
            "applied": False,
            "strategy": "none",
            "query_before": raw_query,
            "query_after": raw_query,
            "inherited_slots": [],
            "overridden_slots": [],
        }

    current = extract_query_intent(raw_query)
    if not _has_recognized_change(current):
        return {
            "applied": False,
            "strategy": "llm_fallback_required",
            "query_before": raw_query,
            "query_after": raw_query,
            "inherited_slots": [],
            "overridden_slots": [],
        }

    inherited: list[str] = []
    overridden: list[str] = []

    if current.get("metrics"):
        metrics = list(current["metrics"])
        overridden.append("metrics")
    else:
        metrics = list(previous_intent.get("metrics") or [])
        if metrics:
            inherited.append("metrics")

    if current.get("dimensions"):
        dimensions = list(current["dimensions"])
        overridden.append("dimensions")
    else:
        dimensions = list(previous_intent.get("dimensions") or [])
        if dimensions:
            inherited.append("dimensions")

    previous_time = dict(previous_intent.get("time") or {})
    current_time = dict(current.get("time") or {})
    # QueryIntent also uses the finest mentioned date part as grain. Only carry
    # grain between turns when the user explicitly said 按天/按月/按年.
    previous_time["grain"] = _explicit_grain(previous_intent.get("query", ""))
    current_time["grain"] = _explicit_grain(raw_query)
    time_info, time_inherited, time_overridden = _merge_time(
        previous_time,
        current_time,
    )
    inherited.extend(time_inherited)
    overridden.extend(time_overridden)

    filters, filter_inherited, filter_overridden = _merge_filters(
        previous_intent.get("filters") or [],
        current.get("filters") or [],
    )
    inherited.extend(filter_inherited)
    overridden.extend(filter_overridden)

    if current.get("top_k") is not None:
        top_k = current["top_k"]
        overridden.append("top_k")
    else:
        top_k = previous_intent.get("top_k")
        if top_k is not None:
            inherited.append("top_k")

    if current.get("order"):
        order = current["order"]
        overridden.append("order")
    else:
        order = previous_intent.get("order")
        if order:
            inherited.append("order")

    merged_intent = {
        "query": raw_query,
        "metrics": metrics,
        "dimensions": dimensions,
        "time": time_info,
        "filters": filters,
        "order": order,
        "top_k": top_k,
    }
    rewritten = render_query_intent(merged_intent)
    if not rewritten or not metrics:
        return {
            "applied": False,
            "strategy": "llm_fallback_required",
            "query_before": raw_query,
            "query_after": raw_query,
            "inherited_slots": [],
            "overridden_slots": [],
        }

    return {
        "applied": True,
        "strategy": "structured_memory",
        "query_before": raw_query,
        "query_after": rewritten,
        "inherited_slots": list(dict.fromkeys(inherited)),
        "overridden_slots": list(dict.fromkeys(overridden)),
    }


def build_turn_input(
    query: str,
    messages: list[dict] | None = None,
) -> dict[str, Any]:
    """Create a clean per-turn input while leaving persisted conversation memory intact."""
    normalized = str(query or "").strip()
    return {
        "query": normalized,
        "raw_query": normalized,
        "messages": messages or [],
        "turn_id": uuid4().hex,
        "turn_started_at": datetime.now(timezone.utc).isoformat(),
        "context_resolution": {},
        "query_intent": {},
        "ambiguity_result": {},
        "clarification_required": False,
        "clarification_question": "",
        "clarification_answer": "",
        "clarification_history": [],
        "clarification_rounds": 0,
        "clarification_cancelled": False,
        "keywords": [],
        "retrieved_columns": [],
        "retrieved_values": [],
        "retrieved_metrics": [],
        "column_recall_sources": {},
        "metric_recall_sources": {},
        "table_infos": [],
        "metric_infos": [],
        "date_info": {},
        "db_info": {},
        "time_semantics": {},
        "metric_semantics": {},
        "sql": "",
        "original_sql": "",
        "audit_result": {},
        "query_plan": {},
        "query_plan_result": {},
        "execution_stats": {},
        "result_summary": {},
        "repair_history": [],
        "repair_guard_result": {},
        "repair_stop_reason": None,
        "error": None,
        "retry_count": 0,
    }
