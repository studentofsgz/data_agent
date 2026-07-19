"""Metrics for Text2SQL offline evaluation."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from eval.schema_linking import aggregate_schema_linking, evaluate_schema_linking


SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_][\w]*)`?",
    re.IGNORECASE,
)

DEFAULT_COLUMN_ALIASES = {
    "gmv": "gmv",
    "total_amount": "gmv",
    "total_sales": "gmv",
    "sales_amount": "gmv",
    "sales_total": "gmv",
    "total_order_amount": "gmv",
    "总销售额": "gmv",
    "销售总额": "gmv",
    "销售额": "gmv",
    "订单总金额": "gmv",
    "aov": "aov",
    "avg_order_amount": "aov",
    "average_order_amount": "aov",
    "客单价": "aov",
    "平均订单金额": "aov",
    "total_quantity": "sales_quantity",
    "sales_quantity": "sales_quantity",
    "total_sales_quantity": "sales_quantity",
    "销量": "sales_quantity",
    "总销量": "sales_quantity",
    "order_count": "order_count",
    "total_orders": "order_count",
    "订单量": "order_count",
    "订单数": "order_count",
}


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).lower()


def extract_tables(sql: str) -> set[str]:
    return {match.group(1).lower() for match in SQL_TABLE_PATTERN.finditer(sql or "")}


def canonical_column_name(
    column_name: Any,
    column_aliases: dict[str, str] | None = None,
) -> str:
    normalized = str(column_name).strip().strip("`").lower()
    aliases = dict(DEFAULT_COLUMN_ALIASES)
    if column_aliases:
        aliases.update(
            {
                str(alias).strip().lower(): str(canonical).strip().lower()
                for alias, canonical in column_aliases.items()
            }
        )
    return aliases.get(normalized, normalized)


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return decimal_value if decimal_value.is_finite() else None


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    numeric = _as_decimal(value)
    if numeric is not None:
        return numeric
    return value


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): serialize_value(value) for key, value in row.items()}
        for row in rows
    ]


def normalize_rows(
    rows: list[dict[str, Any]],
    column_aliases: dict[str, str] | None = None,
    *,
    ordered: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            canonical_key = canonical_column_name(key, column_aliases)
            if canonical_key in normalized_row:
                canonical_key = str(key).strip().lower()
            normalized_row[canonical_key] = normalize_value(value)
        normalized.append(normalized_row)

    if ordered:
        return normalized
    return sorted(normalized, key=lambda row: repr(sorted(row.items())))


def _values_equal(actual: Any, expected: Any, tolerance: Decimal) -> bool:
    actual_number = _as_decimal(actual)
    expected_number = _as_decimal(expected)
    if actual_number is not None and expected_number is not None:
        return abs(actual_number - expected_number) <= tolerance
    return actual == expected


def compare_result_rows(
    rows: list[dict[str, Any]],
    expected_result: list[dict[str, Any]] | None,
    *,
    ordered: bool = False,
    abs_tolerance: float | str | Decimal = 0.01,
    column_aliases: dict[str, str] | None = None,
) -> tuple[bool | None, dict[str, Any] | None]:
    if expected_result is None:
        return None, None

    tolerance = Decimal(str(abs_tolerance))
    actual_rows = normalize_rows(rows, column_aliases, ordered=ordered)
    expected_rows = normalize_rows(expected_result, column_aliases, ordered=ordered)

    if len(actual_rows) != len(expected_rows):
        return False, {
            "reason": "row_count_mismatch",
            "expected_count": len(expected_rows),
            "actual_count": len(actual_rows),
            "expected_preview": serialize_rows(expected_rows[:5]),
            "actual_preview": serialize_rows(actual_rows[:5]),
        }

    for index, (actual_row, expected_row) in enumerate(
        zip(actual_rows, expected_rows, strict=True)
    ):
        actual_columns = set(actual_row)
        expected_columns = set(expected_row)
        if actual_columns != expected_columns:
            return False, {
                "reason": "column_mismatch",
                "row_index": index,
                "expected_columns": sorted(expected_columns),
                "actual_columns": sorted(actual_columns),
            }

        for column in sorted(expected_columns):
            if not _values_equal(actual_row[column], expected_row[column], tolerance):
                return False, {
                    "reason": "value_mismatch",
                    "row_index": index,
                    "column": column,
                    "expected": serialize_value(expected_row[column]),
                    "actual": serialize_value(actual_row[column]),
                    "abs_tolerance": float(tolerance),
                }

    return True, None


def result_matches_expected(
    rows: list[dict[str, Any]],
    expected_result: list[dict[str, Any]] | None,
    *,
    ordered: bool = False,
    abs_tolerance: float | str | Decimal = 0.01,
    column_aliases: dict[str, str] | None = None,
) -> bool | None:
    matched, _ = compare_result_rows(
        rows,
        expected_result,
        ordered=ordered,
        abs_tolerance=abs_tolerance,
        column_aliases=column_aliases,
    )
    return matched


def result_has_data(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and any(
        value is not None
        for row in rows
        for value in row.values()
    )


def sql_rule_matches(sql: str, case: dict[str, Any]) -> bool | None:
    contains = case.get("expect_sql_contains") or []
    forbids = case.get("expect_sql_forbid") or []
    regexes = case.get("expect_sql_regex") or []

    if not contains and not forbids and not regexes:
        return None

    normalized = normalize_sql(sql)
    contains_ok = all(str(token).lower() in normalized for token in contains)
    forbids_ok = all(str(token).lower() not in normalized for token in forbids)
    regex_ok = all(re.search(pattern, normalized, re.IGNORECASE) for pattern in regexes)
    return bool(contains_ok and forbids_ok and regex_ok)


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100, 1)


def metric_block(success: int, total: int) -> dict[str, Any]:
    return {
        "success": success,
        "total": total,
        "rate": rate(success, total),
    }


def optional_metric_block(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    configured = [result for result in results if result[key] is not None]
    return metric_block(
        sum(bool(result[key]) for result in configured),
        len(configured),
    )


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(value, 3)


def timing_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [
        float(item.get("elapsed_seconds") or 0)
        for item in items
    ]
    return {
        "count": len(items),
        "success": sum(item.get("status") == "success" for item in items),
        "error": sum(item.get("status") == "error" for item in items),
        "avg_seconds": round(sum(durations) / len(durations), 3)
        if durations
        else None,
        "p50_seconds": percentile(durations, 0.5),
        "p95_seconds": percentile(durations, 0.95),
        "max_seconds": round(max(durations), 3) if durations else None,
    }


def aggregate_observability(results: list[dict[str, Any]]) -> dict[str, Any]:
    node_items: dict[str, list[dict[str, Any]]] = {}
    llm_items: dict[str, list[dict[str, Any]]] = {}
    all_llm_calls: list[dict[str, Any]] = []

    for result in results:
        for timing in result.get("node_timings") or []:
            node_items.setdefault(str(timing.get("node") or "unknown"), []).append(
                timing
            )
        for call in result.get("llm_calls") or []:
            node = str(call.get("node") or "unknown")
            llm_items.setdefault(node, []).append(call)
            all_llm_calls.append(call)

    usage_reported_calls = [
        call for call in all_llm_calls if call.get("usage_reported")
    ]
    cache_statuses = [
        result.get("sql_cache_status")
        for result in results
        if result.get("sql_cache_status") is not None
    ]

    return {
        "node_timings": {
            node: timing_stats(items)
            for node, items in sorted(node_items.items())
        },
        "llm": {
            **timing_stats(all_llm_calls),
            "total_seconds": round(
                sum(float(call.get("elapsed_seconds") or 0) for call in all_llm_calls),
                3,
            ),
            "input_tokens": sum(
                int(call["input_tokens"])
                for call in usage_reported_calls
                if call.get("input_tokens") is not None
            ),
            "output_tokens": sum(
                int(call["output_tokens"])
                for call in usage_reported_calls
                if call.get("output_tokens") is not None
            ),
            "total_tokens": sum(
                int(call["total_tokens"])
                for call in usage_reported_calls
                if call.get("total_tokens") is not None
            ),
            "usage_reported_calls": len(usage_reported_calls),
            "by_node": {
                node: timing_stats(items)
                for node, items in sorted(llm_items.items())
            },
        },
        "sql_cache": {
            "observed": len(cache_statuses),
            "hits": sum(status == "hit" for status in cache_statuses),
            "misses": sum(status == "miss" for status in cache_statuses),
            "bypassed": sum(status == "bypassed" for status in cache_statuses),
        },
    }


def aggregate_query_governance(results: list[dict[str, Any]]) -> dict[str, Any]:
    plan_events = [
        event
        for result in results
        for event in result.get("query_plan_events") or []
    ]
    sandbox_events = [
        event
        for result in results
        for event in result.get("sql_sandbox_events") or []
    ]
    estimates = [
        int(event.get("estimated_rows") or 0)
        for event in plan_events
    ]
    rejection_codes = Counter(
        str(event.get("code") or "UNKNOWN")
        for event in plan_events
        if event.get("status") == "rejected"
    )
    warning_codes = Counter(
        str(warning)
        for event in plan_events
        for warning in event.get("warnings") or []
    )
    successful_sandbox = [
        event for event in sandbox_events if event.get("status") == "success"
    ]

    return {
        "plan_checks": len(plan_events),
        "plan_passed": sum(event.get("status") == "passed" for event in plan_events),
        "plan_rejected": sum(event.get("status") == "rejected" for event in plan_events),
        "rejection_codes": dict(sorted(rejection_codes.items())),
        "warning_codes": dict(sorted(warning_codes.items())),
        "avg_estimated_rows": round(sum(estimates) / len(estimates), 1)
        if estimates
        else None,
        "max_estimated_rows": max(estimates) if estimates else None,
        "sandbox_executions": len(successful_sandbox),
        "sandbox_timeouts": sum(
            event.get("status") == "timeout" for event in sandbox_events
        ),
        "sandbox_truncated": sum(
            bool(event.get("truncated")) for event in successful_sandbox
        ),
        "sandbox_returned_rows": sum(
            int(event.get("returned_rows") or 0) for event in successful_sandbox
        ),
    }


def aggregate_clarification(results: list[dict[str, Any]]) -> dict[str, Any]:
    configured = [
        result
        for result in results
        if result.get("clarification_expected") is not None
    ]
    true_positive = sum(
        result["clarification_expected"] is True
        and result["clarification_required"] is True
        for result in configured
    )
    true_negative = sum(
        result["clarification_expected"] is False
        and result["clarification_required"] is False
        for result in configured
    )
    false_positive = sum(
        result["clarification_expected"] is False
        and result["clarification_required"] is True
        for result in configured
    )
    false_negative = sum(
        result["clarification_expected"] is True
        and result["clarification_required"] is False
        for result in configured
    )
    precision_total = true_positive + false_positive
    recall_total = true_positive + false_negative
    return {
        "configured_cases": len(configured),
        "accuracy": metric_block(
            sum(bool(result.get("clarification_ok")) for result in configured),
            len(configured),
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": rate(true_positive, precision_total),
        "recall": rate(true_positive, recall_total),
        "unnecessary_clarification_rate": rate(false_positive, len(configured)),
    }


def evaluate_case(
    *,
    case: dict[str, Any],
    sql: str,
    rows: list[dict[str, Any]],
    error: str,
    elapsed_seconds: float,
    correction_attempts: int,
    event_count: int,
    result_received: bool,
    node_timings: list[dict[str, Any]] | None = None,
    llm_calls: list[dict[str, Any]] | None = None,
    sql_cache_hit: bool | None = None,
    sql_cache_status: str | None = None,
    repair_guard_events: list[dict[str, Any]] | None = None,
    schema_linking_events: list[dict[str, Any]] | None = None,
    query_plan_events: list[dict[str, Any]] | None = None,
    sql_sandbox_events: list[dict[str, Any]] | None = None,
    query_intent_event: dict[str, Any] | None = None,
    clarification_event: dict[str, Any] | None = None,
    context_resolution_event: dict[str, Any] | None = None,
    conversation_memory_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_tables = {str(t).lower() for t in case.get("expect_tables", [])}
    actual_tables = extract_tables(sql)
    expected_result = case.get("expected_result")

    sql_generated = bool(sql.strip())
    sql_executable = bool(result_received and not error)

    expect_not_empty = case.get("expect_not_empty")
    not_empty_ok: bool | None = None
    if isinstance(expect_not_empty, bool):
        has_data = result_has_data(rows)
        not_empty_ok = has_data if expect_not_empty else not has_data

    expected_tables_hit = True
    if expected_tables:
        expected_tables_hit = expected_tables.issubset(actual_tables)

    expected_result_ok, expected_result_diff = compare_result_rows(
        rows,
        expected_result,
        ordered=bool(case.get("expected_result_ordered", False)),
        abs_tolerance=case.get("result_abs_tolerance", 0.01),
        column_aliases=case.get("result_column_aliases"),
    )
    expected_sql_rule_ok = sql_rule_matches(sql, case)
    node_timings = list(node_timings or [])
    llm_calls = list(llm_calls or [])
    repair_guard_events = list(repair_guard_events or [])
    schema_linking_events = list(schema_linking_events or [])
    query_plan_events = list(query_plan_events or [])
    sql_sandbox_events = list(sql_sandbox_events or [])
    clarification_required = clarification_event is not None
    clarification_expected = case.get("expect_clarification")
    expected_clarification_code = case.get("expect_clarification_code")
    actual_clarification_code = (
        clarification_event.get("code") if clarification_event else None
    )
    clarification_ok: bool | None = None
    if isinstance(clarification_expected, bool):
        clarification_ok = clarification_required == clarification_expected
        if clarification_ok and expected_clarification_code is not None:
            clarification_ok = (
                actual_clarification_code == expected_clarification_code
            )
    slowest_node = (
        max(
            node_timings,
            key=lambda item: float(item.get("elapsed_seconds") or 0),
        )
        if node_timings
        else None
    )
    usage_reported_calls = [
        call for call in llm_calls if call.get("usage_reported")
    ]
    if sql_cache_status is None:
        if sql_cache_hit is True:
            sql_cache_status = "hit"
        elif sql_cache_hit is False:
            sql_cache_status = "miss"

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "difficulty": case.get("difficulty", "unknown"),
        "category": case.get("category", "unknown"),
        "sql": sql,
        "gold_sql": case.get("gold_sql"),
        "normalized_sql": normalize_sql(sql),
        "actual_tables": sorted(actual_tables),
        "expected_tables": sorted(expected_tables),
        "sql_generated": sql_generated,
        "sql_executable": sql_executable,
        "expected_tables_hit": expected_tables_hit,
        "not_empty_ok": not_empty_ok,
        "expected_result_ok": expected_result_ok,
        "expected_result_diff": expected_result_diff,
        "expected_result": expected_result,
        "actual_result": serialize_rows(rows) if expected_result is not None else None,
        "expected_sql_rule_ok": expected_sql_rule_ok,
        "result_count": len(rows),
        "elapsed_seconds": elapsed_seconds,
        "correction_attempts": correction_attempts,
        "event_count": event_count,
        "node_timings": node_timings,
        "node_timing_count": len(node_timings),
        "slowest_node": {
            "node": slowest_node.get("node"),
            "elapsed_seconds": slowest_node.get("elapsed_seconds"),
        }
        if slowest_node
        else None,
        "llm_calls": llm_calls,
        "llm_call_count": len(llm_calls),
        "llm_elapsed_seconds": round(
            sum(float(call.get("elapsed_seconds") or 0) for call in llm_calls),
            3,
        ),
        "llm_input_tokens": sum(
            int(call["input_tokens"])
            for call in usage_reported_calls
            if call.get("input_tokens") is not None
        ),
        "llm_output_tokens": sum(
            int(call["output_tokens"])
            for call in usage_reported_calls
            if call.get("output_tokens") is not None
        ),
        "llm_total_tokens": sum(
            int(call["total_tokens"])
            for call in usage_reported_calls
            if call.get("total_tokens") is not None
        ),
        "llm_usage_reported_calls": len(usage_reported_calls),
        "sql_cache_status": sql_cache_status,
        "sql_cache_hit": True
        if sql_cache_status == "hit"
        else False
        if sql_cache_status == "miss"
        else None,
        "repair_guard_events": repair_guard_events,
        "repair_stop_reason": next(
            (
                str(event.get("code") or "UNKNOWN")
                for event in reversed(repair_guard_events)
                if event.get("status") == "stopped"
            ),
            None,
        ),
        "schema_linking_events": schema_linking_events,
        "schema_linking": evaluate_schema_linking(case, schema_linking_events),
        "query_plan_events": query_plan_events,
        "sql_sandbox_events": sql_sandbox_events,
        "query_intent_event": query_intent_event,
        "context_resolution_event": context_resolution_event,
        "conversation_memory_event": conversation_memory_event,
        "clarification_event": clarification_event,
        "clarification_required": clarification_required,
        "clarification_expected": clarification_expected
        if isinstance(clarification_expected, bool)
        else None,
        "clarification_code": actual_clarification_code,
        "clarification_ok": clarification_ok,
        "error": error,
    }


def aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    repair_stop_reasons = Counter(
        str(result["repair_stop_reason"])
        for result in results
        if result.get("repair_stop_reason")
    )
    return {
        "total": total,
        "sql_generated": metric_block(sum(r["sql_generated"] for r in results), total),
        "sql_executable": metric_block(sum(r["sql_executable"] for r in results), total),
        "expected_tables_hit": metric_block(sum(r["expected_tables_hit"] for r in results), total),
        "not_empty_ok": optional_metric_block(results, "not_empty_ok"),
        "expected_result_ok": optional_metric_block(results, "expected_result_ok"),
        "expected_sql_rule_ok": optional_metric_block(results, "expected_sql_rule_ok"),
        "self_repair_cases": sum(1 for r in results if r["correction_attempts"] > 0),
        "repair_guard_stopped_cases": sum(repair_stop_reasons.values()),
        "repair_stop_reasons": dict(sorted(repair_stop_reasons.items())),
        "avg_seconds": round(sum(r["elapsed_seconds"] for r in results) / total, 3)
        if total
        else 0.0,
        "observability": aggregate_observability(results),
        "schema_linking": aggregate_schema_linking(results),
        "query_governance": aggregate_query_governance(results),
        "clarification": aggregate_clarification(results),
    }


def aggregate_by(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        group = str(item.get(key) or "unknown")
        groups.setdefault(group, []).append(item)
    return {
        group_key: aggregate_group(group_items)
        for group_key, group_items in sorted(groups.items())
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate_group(results)
    summary["by_difficulty"] = aggregate_by(results, "difficulty")
    summary["by_category"] = aggregate_by(results, "category")
    return {"summary": summary}
