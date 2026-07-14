"""Metrics for Text2SQL offline evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_][\w]*)`?",
    re.IGNORECASE,
)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).lower()


def extract_tables(sql: str) -> set[str]:
    return {match.group(1).lower() for match in SQL_TABLE_PATTERN.finditer(sql or "")}


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {str(k): normalize_value(v) for k, v in row.items()}
        for row in rows
    ]
    return sorted(normalized, key=lambda row: repr(sorted(row.items())))


def result_matches_expected(
    rows: list[dict[str, Any]],
    expected_result: list[dict[str, Any]] | None,
) -> bool | None:
    if expected_result is None:
        return None
    return normalize_rows(rows) == normalize_rows(expected_result)


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


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


def metric_block(success: int, total: int) -> dict[str, Any]:
    return {
        "success": success,
        "total": total,
        "rate": rate(success, total),
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
) -> dict[str, Any]:
    expected_tables = {str(t).lower() for t in case.get("expect_tables", [])}
    actual_tables = extract_tables(sql)
    expected_result = case.get("expected_result")

    sql_generated = bool(sql.strip())
    sql_executable = bool(result_received and not error)
    not_empty_ok = True
    if case.get("expect_not_empty"):
        not_empty_ok = bool(rows)

    expected_tables_hit = True
    if expected_tables:
        expected_tables_hit = expected_tables.issubset(actual_tables)

    expected_result_ok = result_matches_expected(rows, expected_result)
    expected_sql_rule_ok = sql_rule_matches(sql, case)

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "difficulty": case.get("difficulty", "unknown"),
        "category": case.get("category", "unknown"),
        "sql": sql,
        "normalized_sql": normalize_sql(sql),
        "actual_tables": sorted(actual_tables),
        "expected_tables": sorted(expected_tables),
        "sql_generated": sql_generated,
        "sql_executable": sql_executable,
        "expected_tables_hit": expected_tables_hit,
        "not_empty_ok": not_empty_ok,
        "expected_result_ok": expected_result_ok,
        "expected_sql_rule_ok": expected_sql_rule_ok,
        "result_count": len(rows),
        "elapsed_seconds": elapsed_seconds,
        "correction_attempts": correction_attempts,
        "event_count": event_count,
        "error": error,
    }


def aggregate_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    expected_result_cases = [r for r in results if r["expected_result_ok"] is not None]
    expected_sql_rule_cases = [r for r in results if r["expected_sql_rule_ok"] is not None]
    return {
        "total": total,
        "sql_generated": metric_block(sum(r["sql_generated"] for r in results), total),
        "sql_executable": metric_block(sum(r["sql_executable"] for r in results), total),
        "expected_tables_hit": metric_block(sum(r["expected_tables_hit"] for r in results), total),
        "not_empty_ok": metric_block(sum(r["not_empty_ok"] for r in results), total),
        "expected_result_ok": metric_block(
            sum(r["expected_result_ok"] for r in expected_result_cases),
            len(expected_result_cases),
        ),
        "expected_sql_rule_ok": metric_block(
            sum(r["expected_sql_rule_ok"] for r in expected_sql_rule_cases),
            len(expected_sql_rule_cases),
        ),
        "self_repair_cases": sum(1 for r in results if r["correction_attempts"] > 0),
        "avg_seconds": round(sum(r["elapsed_seconds"] for r in results) / total, 3)
        if total
        else 0.0,
    }


def aggregate_by(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get(key) or "unknown")].append(item)
    return {
        group_key: aggregate_group(group_items)
        for group_key, group_items in sorted(groups.items())
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate_group(results)
    summary["by_difficulty"] = aggregate_by(results, "difficulty")
    summary["by_category"] = aggregate_by(results, "category")
    return {"summary": summary}
