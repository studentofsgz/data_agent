"""Pure helpers for bounded, traceable and numerically grounded answers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


NUMBER_PATTERN = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


def _safe_cell(value: Any, max_chars: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def sanitize_answer_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    max_cell_chars: int,
) -> list[dict[str, Any]]:
    """Bound model context and convert database-specific values to JSON-safe data."""
    return [
        {
            str(column): _safe_cell(value, max(1, max_cell_chars))
            for column, value in row.items()
        }
        for row in rows[: max(0, max_rows)]
    ]


def _to_decimal(token: str) -> Decimal | None:
    try:
        value = Decimal(token.replace(",", ""))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def _number_tokens(value: Any) -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float, Decimal)):
        return [str(value)]
    if isinstance(value, dict):
        return [
            token
            for item in value.values()
            for token in _number_tokens(item)
        ]
    if isinstance(value, (list, tuple)):
        return [token for item in value for token in _number_tokens(item)]
    return NUMBER_PATTERN.findall(str(value))


def validate_numeric_grounding(
    *,
    texts: list[str],
    rows: list[dict[str, Any]],
    query: str,
    row_count: int,
    tolerance: float,
) -> dict[str, Any]:
    """Reject numeric claims that are absent from result rows or the user query."""
    answer_tokens = list(dict.fromkeys(
        token for text in texts for token in NUMBER_PATTERN.findall(text)
    ))
    allowed_tokens = list(dict.fromkeys([
        *_number_tokens(rows),
        *_number_tokens(query),
        str(row_count),
    ]))
    allowed_values = [
        value
        for token in allowed_tokens
        if (value := _to_decimal(token)) is not None
    ]
    decimal_tolerance = Decimal(str(max(0.0, tolerance)))
    invalid = []
    for token in answer_tokens:
        value = _to_decimal(token)
        if value is None or not any(
            abs(value - allowed) <= decimal_tolerance
            for allowed in allowed_values
        ):
            invalid.append(token)
    return {
        "passed": not invalid,
        "answer_numbers": answer_tokens,
        "allowed_numbers": allowed_tokens,
        "invalid_numbers": invalid,
    }


def deterministic_answer(
    rows: list[dict[str, Any]],
    *,
    row_count: int,
    max_rows: int,
) -> str:
    """Build a safe fallback that copies values without analytical inference."""
    if row_count == 0 or not rows:
        return "没有查询到符合条件的数据。"
    shown = rows[: max(1, max_rows)]
    row_texts = [
        "，".join(f"{column}={value}" for column, value in row.items())
        for row in shown
    ]
    prefix = f"本次查询返回{row_count}行。"
    if len(shown) < row_count:
        return prefix + f"前{len(shown)}行是：" + "；".join(row_texts) + "。"
    return prefix + "结果是：" + "；".join(row_texts) + "。"


def build_provenance(state: dict[str, Any], rows_used: int) -> dict[str, Any]:
    audit = state.get("audit_result") or {}
    summary = state.get("result_summary") or {}
    semantic_metrics = [
        {
            "name": item.get("name"),
            "display_name": item.get("display_name"),
            "expression": item.get("expression"),
        }
        for item in (state.get("metric_semantics") or {}).get("metrics") or []
    ]
    if not semantic_metrics:
        semantic_metrics = [
            {
                "name": item.get("name"),
                "display_name": item.get("description"),
                "expression": item.get("expression"),
            }
            for item in state.get("metric_infos") or []
        ]
    intent = state.get("query_intent") or {}
    return {
        "resolved_query": state.get("query", ""),
        "sql": state.get("sql", ""),
        "tables": list(audit.get("tables") or []),
        "columns": list(audit.get("columns") or []),
        "metrics": semantic_metrics,
        "filters": list(intent.get("filters") or []),
        "time": dict(intent.get("time") or {}),
        "row_count": int(summary.get("row_count") or 0),
        "rows_used": rows_used,
        "truncated": bool(summary.get("truncated")),
    }


def system_caveats(provenance: dict[str, Any]) -> list[str]:
    caveats = []
    if provenance["truncated"]:
        caveats.append("查询结果已达到执行上限，回答仅基于已返回的数据。")
    if provenance["rows_used"] < provenance["row_count"]:
        caveats.append(
            f"自然语言回答仅使用前{provenance['rows_used']}行，完整数据请查看查询结果。"
        )
    return caveats
