"""Pure MySQL EXPLAIN JSON parsing and query-cost policy checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


@dataclass(frozen=True)
class QueryPlanPolicy:
    max_estimated_rows: int
    max_full_scan_rows: int
    max_join_tables: int
    reject_cartesian_joins: bool = True


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    number = _as_number(value)
    return max(0, int(number)) if number is not None else 0


def _normalize_plan(plan: Any) -> dict[str, Any]:
    if isinstance(plan, str):
        parsed = json.loads(plan)
        if not isinstance(parsed, dict):
            raise ValueError("EXPLAIN JSON根节点必须是对象")
        return parsed
    if not isinstance(plan, dict):
        raise ValueError("EXPLAIN结果不是JSON对象")
    return plan


def _has_flag(value: Any, flag: str) -> bool:
    if isinstance(value, dict):
        if value.get(flag) is True:
            return True
        return any(_has_flag(child, flag) for child in value.values())
    if isinstance(value, list):
        return any(_has_flag(child, flag) for child in value)
    return False


def _collect_tables(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, list):
        for child in value:
            _collect_tables(child, output)
        return
    if not isinstance(value, dict):
        return

    table = value.get("table")
    if isinstance(table, dict) and table.get("table_name"):
        examined_rows = _as_int(table.get("rows_examined_per_scan"))
        produced_rows = _as_int(table.get("rows_produced_per_join"))
        output.append({
            "table": str(table.get("table_name")),
            "access_type": str(table.get("access_type") or "UNKNOWN").upper(),
            "estimated_rows": max(examined_rows, produced_rows),
            "rows_examined_per_scan": examined_rows,
            "rows_produced_per_join": produced_rows,
            "key": table.get("key"),
            "possible_keys": list(table.get("possible_keys") or []),
        })

    for key, child in value.items():
        if key != "table":
            _collect_tables(child, output)


def _query_cost(value: Any) -> float | None:
    costs: list[float] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            cost_info = node.get("cost_info")
            if isinstance(cost_info, dict):
                cost = _as_number(cost_info.get("query_cost"))
                if cost is not None:
                    costs.append(cost)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return round(max(costs), 3) if costs else None


def _has_cartesian_join(sql: str, dialect: str) -> bool:
    try:
        expression = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, ValueError):
        return False

    for join in expression.find_all(exp.Join):
        kind = str(join.args.get("kind") or "").upper()
        method = str(join.args.get("method") or "").upper()
        if kind == "CROSS":
            return True
        if method != "NATURAL" and not join.args.get("on") and not join.args.get("using"):
            return True
    return False


def evaluate_query_plan(
    *,
    sql: str,
    plan: Any,
    policy: QueryPlanPolicy,
    dialect: str = "mysql",
) -> dict[str, Any]:
    """Return a stable, serializable cost-guard decision without database I/O."""
    try:
        normalized_plan = _normalize_plan(plan)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "passed": False,
            "code": "INVALID_QUERY_PLAN",
            "message": "无法解析数据库执行计划，拒绝执行",
            "estimated_rows": 0,
            "query_cost": None,
            "join_table_count": 0,
            "tables": [],
            "full_scan_tables": [],
            "warnings": [],
            "violations": ["INVALID_QUERY_PLAN"],
            "details": {"error": str(exc)[:500]},
        }

    tables: list[dict[str, Any]] = []
    _collect_tables(normalized_plan, tables)
    estimated_rows = max(
        (int(table["estimated_rows"]) for table in tables),
        default=0,
    )
    full_scan_tables = [
        str(table["table"])
        for table in tables
        if table["access_type"] == "ALL"
        and int(table["rows_examined_per_scan"]) >= policy.max_full_scan_rows
    ]
    cartesian_join = _has_cartesian_join(sql, dialect)

    warnings: list[str] = []
    if _has_flag(normalized_plan, "using_temporary_table"):
        warnings.append("USING_TEMPORARY_TABLE")
    if _has_flag(normalized_plan, "using_filesort"):
        warnings.append("USING_FILESORT")

    violations: list[str] = []
    if policy.reject_cartesian_joins and cartesian_join:
        violations.append("CARTESIAN_JOIN")
    if len(tables) > policy.max_join_tables:
        violations.append("JOIN_TABLE_LIMIT_EXCEEDED")
    if full_scan_tables:
        violations.append("LARGE_FULL_SCAN")
    if estimated_rows > policy.max_estimated_rows:
        violations.append("ESTIMATED_ROWS_LIMIT_EXCEEDED")

    messages = {
        "CARTESIAN_JOIN": "检测到笛卡尔积或缺少连接条件，拒绝执行",
        "JOIN_TABLE_LIMIT_EXCEEDED": "查询关联表数量超过成本策略上限，拒绝执行",
        "LARGE_FULL_SCAN": "查询会对大表进行全表扫描，拒绝执行",
        "ESTIMATED_ROWS_LIMIT_EXCEEDED": "查询预计处理行数超过成本策略上限，拒绝执行",
    }
    code = violations[0] if violations else "PLAN_OK"
    return {
        "passed": not violations,
        "code": code,
        "message": messages.get(code, "查询执行计划成本检查通过"),
        "estimated_rows": estimated_rows,
        "query_cost": _query_cost(normalized_plan),
        "join_table_count": len(tables),
        "tables": tables,
        "full_scan_tables": full_scan_tables,
        "warnings": warnings,
        "violations": violations,
        "details": {
            "cartesian_join": cartesian_join,
            "limits": {
                "max_estimated_rows": policy.max_estimated_rows,
                "max_full_scan_rows": policy.max_full_scan_rows,
                "max_join_tables": policy.max_join_tables,
            },
        },
    }
