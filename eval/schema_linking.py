"""Schema Linking expectations and retrieval-stage metrics."""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


EMPTY_EXPECTATIONS = {"tables": [], "columns": [], "metrics": [], "join_keys": []}


def _canonical_set(values: list[Any] | set[Any] | None) -> set[str]:
    return {
        str(value).strip().strip("`").casefold()
        for value in values or []
        if str(value).strip()
    }


def _qualified_column(column: exp.Column, aliases: dict[str, str], tables: set[str]) -> str | None:
    column_name = str(column.name or "").casefold()
    qualifier = str(column.table or "").casefold()
    if not column_name:
        return None
    if qualifier:
        table_name = aliases.get(qualifier)
        return f"{table_name}.{column_name}" if table_name else None
    if len(tables) == 1:
        return f"{next(iter(tables))}.{column_name}"
    return None


def _projection_metrics(projection: exp.Expression) -> set[str]:
    nodes = list(projection.walk())
    sum_amount = any(
        isinstance(node, exp.Sum)
        and any(
            isinstance(child, exp.Column) and str(child.name).casefold() == "order_amount"
            for child in node.walk()
        )
        for node in nodes
    )
    sum_quantity = any(
        isinstance(node, exp.Sum)
        and any(
            isinstance(child, exp.Column) and str(child.name).casefold() == "order_quantity"
            for child in node.walk()
        )
        for node in nodes
    )
    count_distinct_order = any(
        isinstance(node, exp.Count)
        and node.find(exp.Distinct) is not None
        and any(
            isinstance(child, exp.Column) and str(child.name).casefold() == "order_id"
            for child in node.walk()
        )
        for node in nodes
    )

    metrics: set[str] = set()
    if sum_amount and count_distinct_order:
        metrics.add("aov")
    elif sum_amount:
        metrics.add("gmv")
    elif count_distinct_order:
        metrics.add("order_count")
    if sum_quantity:
        metrics.add("sales_quantity")
    return metrics


def derive_gold_schema(gold_sql: str, dialect: str = "mysql") -> dict[str, list[str]]:
    """Derive reviewed tables, columns, metrics and JOIN keys from a golden SQL."""
    if not (gold_sql or "").strip():
        return {key: list(value) for key, value in EMPTY_EXPECTATIONS.items()}
    try:
        expression = sqlglot.parse_one(gold_sql, read=dialect)
    except (ParseError, ValueError):
        return {key: list(value) for key, value in EMPTY_EXPECTATIONS.items()}

    tables: set[str] = set()
    aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        table_name = str(table.name or "").casefold()
        if not table_name:
            continue
        tables.add(table_name)
        aliases[table_name] = table_name
        if table.alias_or_name:
            aliases[str(table.alias_or_name).casefold()] = table_name

    projection_aliases = {
        str(projection.alias).casefold()
        for select in expression.find_all(exp.Select)
        for projection in select.expressions
        if projection.alias
    }
    columns: set[str] = set()
    for column in expression.find_all(exp.Column):
        if not column.table and str(column.name or "").casefold() in projection_aliases:
            continue
        qualified = _qualified_column(column, aliases, tables)
        if qualified:
            columns.add(qualified)

    join_keys: set[str] = set()
    for join in expression.find_all(exp.Join):
        on_expression = join.args.get("on")
        if on_expression is None:
            continue
        for predicate in on_expression.walk():
            if not isinstance(predicate, exp.EQ):
                continue
            left = predicate.this
            right = predicate.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            left_id = _qualified_column(left, aliases, tables)
            right_id = _qualified_column(right, aliases, tables)
            if left_id and right_id and left_id != right_id:
                join_keys.add("=".join(sorted((left_id, right_id))))

    metrics: set[str] = set()
    for select in expression.find_all(exp.Select):
        for projection in select.expressions:
            metrics.update(_projection_metrics(projection))

    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
        "metrics": sorted(metrics),
        "join_keys": sorted(join_keys),
    }


def _last_stage(events: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return next(
        (event for event in reversed(events) if event.get("stage") == stage),
        {},
    )


def _recall_block(expected: set[str], actual: set[str]) -> dict[str, Any]:
    if not expected:
        return {"success": 0, "total": 0, "rate": None, "exact": None}
    success = len(expected & actual)
    return {
        "success": success,
        "total": len(expected),
        "rate": round(success / len(expected) * 100, 1),
        "exact": expected.issubset(actual),
    }


def evaluate_schema_linking(
    case: dict[str, Any],
    events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    events = list(events or [])
    derived = derive_gold_schema(str(case.get("gold_sql") or ""))
    expected_tables = _canonical_set(case.get("expect_tables") or derived["tables"])
    expected_columns = _canonical_set(case.get("expect_columns") or derived["columns"])
    expected_metrics = _canonical_set(case.get("expect_metrics") or derived["metrics"])
    expected_join_keys = _canonical_set(case.get("expect_join_keys") or derived["join_keys"])

    column_recall = _last_stage(events, "column_recall")
    metric_recall = _last_stage(events, "metric_recall")
    rerank = _last_stage(events, "rerank")
    merge = _last_stage(events, "merge")
    table_filter = _last_stage(events, "table_filter")
    metric_filter = _last_stage(events, "metric_filter")

    raw_columns = _canonical_set([
        candidate.get("id") for candidate in column_recall.get("candidates") or []
    ])
    reranked_columns = _canonical_set([
        candidate.get("id") for candidate in rerank.get("columns") or []
    ])
    final_columns = _canonical_set(table_filter.get("columns") or merge.get("columns") or [])
    final_tables = _canonical_set(table_filter.get("tables") or merge.get("tables") or [])
    raw_metrics = _canonical_set([
        candidate.get("id") for candidate in metric_recall.get("candidates") or []
    ])
    reranked_metrics = _canonical_set([
        candidate.get("id") for candidate in rerank.get("metrics") or []
    ])
    final_metrics = _canonical_set(metric_filter.get("metrics") or merge.get("metrics") or [])

    source_columns: dict[str, set[str]] = {}
    for candidate in column_recall.get("candidates") or []:
        candidate_id = str(candidate.get("id") or "").casefold()
        for source in candidate.get("sources") or []:
            source_columns.setdefault(str(source), set()).add(candidate_id)

    source_metrics: dict[str, set[str]] = {}
    for candidate in metric_recall.get("candidates") or []:
        candidate_id = str(candidate.get("id") or "").casefold()
        for source in candidate.get("sources") or []:
            source_metrics.setdefault(str(source), set()).add(candidate_id)

    covered_join_keys = {
        join_key
        for join_key in expected_join_keys
        if set(join_key.split("=", 1)).issubset(final_columns)
    }
    return {
        "observed": bool(events),
        "expected": {
            "tables": sorted(expected_tables),
            "columns": sorted(expected_columns),
            "metrics": sorted(expected_metrics),
            "join_keys": sorted(expected_join_keys),
        },
        "actual": {
            "tables": sorted(final_tables),
            "columns": sorted(final_columns),
            "metrics": sorted(final_metrics),
            "raw_columns": sorted(raw_columns),
            "reranked_columns": sorted(reranked_columns),
            "raw_metrics": sorted(raw_metrics),
            "reranked_metrics": sorted(reranked_metrics),
        },
        "table_recall": _recall_block(expected_tables, final_tables),
        "raw_column_recall": _recall_block(expected_columns, raw_columns),
        "column_recall_at_k": _recall_block(expected_columns, reranked_columns),
        "final_column_recall": _recall_block(expected_columns, final_columns),
        "raw_metric_recall": _recall_block(expected_metrics, raw_metrics),
        "metric_recall_at_k": _recall_block(expected_metrics, reranked_metrics),
        "final_metric_recall": _recall_block(expected_metrics, final_metrics),
        "join_key_coverage": _recall_block(expected_join_keys, covered_join_keys),
        "column_recall_by_source": {
            source: _recall_block(expected_columns, candidates)
            for source, candidates in sorted(source_columns.items())
        },
        "metric_recall_by_source": {
            source: _recall_block(expected_metrics, candidates)
            for source, candidates in sorted(source_metrics.items())
        },
        "missing": {
            "tables": sorted(expected_tables - final_tables),
            "columns": sorted(expected_columns - final_columns),
            "metrics": sorted(expected_metrics - final_metrics),
            "join_keys": sorted(expected_join_keys - covered_join_keys),
        },
        "candidate_counts": {
            "raw_columns": len(raw_columns),
            "reranked_columns": len(reranked_columns),
            "final_columns": len(final_columns),
            "raw_metrics": len(raw_metrics),
            "reranked_metrics": len(reranked_metrics),
            "final_metrics": len(final_metrics),
        },
    }


def _aggregate_blocks(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    success = sum(int(item[key]["success"]) for item in items)
    total = sum(int(item[key]["total"]) for item in items)
    exact_cases = [item[key]["exact"] for item in items if item[key]["exact"] is not None]
    return {
        "success": success,
        "total": total,
        "rate": round(success / total * 100, 1) if total else None,
        "exact_cases": sum(bool(value) for value in exact_cases),
        "configured_cases": len(exact_cases),
    }


def aggregate_schema_linking(results: list[dict[str, Any]]) -> dict[str, Any]:
    items = [
        result["schema_linking"]
        for result in results
        if (result.get("schema_linking") or {}).get("observed")
    ]
    metric_keys = (
        "table_recall",
        "raw_column_recall",
        "column_recall_at_k",
        "final_column_recall",
        "raw_metric_recall",
        "metric_recall_at_k",
        "final_metric_recall",
        "join_key_coverage",
    )
    empty_block = {
        "success": 0,
        "total": 0,
        "rate": None,
        "exact_cases": 0,
        "configured_cases": 0,
    }
    column_sources = sorted({
        source
        for item in items
        for source in item.get("column_recall_by_source") or {}
    })
    column_source_summary = {}
    for source in column_sources:
        source_items = [
            {
                "source": item["column_recall_by_source"].get(
                    source,
                    _recall_block(_canonical_set(item["expected"]["columns"]), set()),
                )
            }
            for item in items
        ]
        column_source_summary[source] = _aggregate_blocks(source_items, "source")

    metric_sources = sorted({
        source
        for item in items
        for source in item.get("metric_recall_by_source") or {}
    })
    metric_source_summary = {}
    for source in metric_sources:
        source_items = [
            {
                "source": item["metric_recall_by_source"].get(
                    source,
                    _recall_block(_canonical_set(item["expected"]["metrics"]), set()),
                )
            }
            for item in items
        ]
        metric_source_summary[source] = _aggregate_blocks(source_items, "source")

    candidate_keys = (
        "raw_columns",
        "reranked_columns",
        "final_columns",
        "raw_metrics",
        "reranked_metrics",
        "final_metrics",
    )
    return {
        "observed_cases": len(items),
        **{
            key: _aggregate_blocks(items, key) if items else dict(empty_block)
            for key in metric_keys
        },
        "column_recall_by_source": column_source_summary,
        "metric_recall_by_source": metric_source_summary,
        "avg_candidate_counts": {
            key: round(
                sum(item["candidate_counts"][key] for item in items) / len(items),
                2,
            )
            if items
            else None
            for key in candidate_keys
        },
    }
