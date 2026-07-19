import json
from collections import Counter
from typing import Any

import sqlglot
from langgraph.runtime import Runtime
from sqlglot import exp
from sqlglot.errors import ParseError

from app.agent.context import DataAgentContext
from app.agent.state import (
    DataAgentState,
    SQLRepairGuardResultState,
)
from app.core.log import logger


def _parse_single_query(sql: str, dialect: str) -> exp.Query | None:
    try:
        statements = [
            statement
            for statement in sqlglot.parse((sql or "").strip(), read=dialect)
            if statement is not None
        ]
    except (ParseError, ValueError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None
    return statements[0]


def sql_fingerprint(sql: str, dialect: str = "mysql") -> str:
    """Return a formatting-insensitive SQL identity used for loop detection."""
    expression = _parse_single_query(sql, dialect)
    if expression is None:
        return " ".join((sql or "").casefold().split()).rstrip(";")
    return expression.sql(dialect=dialect, pretty=False, normalize=True).rstrip(";")


def _filter_nodes(expression: exp.Query) -> list[exp.Expression]:
    nodes: list[exp.Expression] = []
    nodes.extend(expression.find_all(exp.Where))
    nodes.extend(expression.find_all(exp.Having))
    nodes.extend(expression.find_all(exp.Qualify))
    return nodes


def _literal_counter(nodes: list[exp.Expression], dialect: str) -> Counter[str]:
    return Counter(
        literal.sql(dialect=dialect, normalize=True)
        for node in nodes
        for literal in node.find_all(exp.Literal)
    )


def _semantic_signature(expression: exp.Query, dialect: str) -> dict[str, Any]:
    filters = _filter_nodes(expression)
    joins = list(expression.find_all(exp.Join))
    limits = list(expression.find_all(exp.Limit))
    literal_limit = None
    if limits:
        raw_limit = limits[-1].expression
        if isinstance(raw_limit, exp.Literal) and not raw_limit.is_string:
            literal_limit = str(raw_limit.this)

    aggregates = Counter(
        f"{aggregate.key}:{'distinct' if aggregate.find(exp.Distinct) else 'all'}"
        for aggregate in expression.find_all(exp.AggFunc)
    )
    groups = list(expression.find_all(exp.Group))
    orders = list(expression.find_all(exp.Ordered))

    return {
        "aggregates": dict(sorted(aggregates.items())),
        "join_count": sum(1 for _ in expression.find_all(exp.Join)),
        "join_types": dict(sorted(Counter(
            f"{str(join.args.get('side') or 'INNER').upper()}:"
            f"{str(join.args.get('kind') or 'JOIN').upper()}"
            for join in joins
        ).items())),
        "join_predicate_count": sum(
            1
            for join in joins
            for node in (join.args.get("on").walk() if join.args.get("on") else [])
            if isinstance(node, exp.Predicate)
        ),
        "group_expression_count": sum(len(group.expressions) for group in groups),
        "filter_clause_count": len(filters),
        "filter_predicate_count": sum(
            1 for node in filters for _ in node.find_all(exp.Predicate)
        ),
        "filter_literals": dict(sorted(_literal_counter(filters, dialect).items())),
        "filter_operators": dict(sorted(Counter(
            node.key
            for filter_node in filters
            for node in filter_node.walk()
            if isinstance(node, exp.Predicate)
        ).items())),
        "filter_boolean_operators": dict(sorted(Counter(
            node.key
            for filter_node in filters
            for node in filter_node.walk()
            if isinstance(node, (exp.And, exp.Or, exp.Not))
        ).items())),
        "select_distinct_count": sum(
            1
            for select in expression.find_all(exp.Select)
            if select.args.get("distinct") is not None
        ),
        "set_operation_count": sum(
            1 for _ in expression.find_all(exp.SetOperation)
        ),
        "order_expression_count": len(orders),
        "descending_order_count": sum(
            1 for ordered in orders if ordered.args.get("desc") is True
        ),
        "literal_limit": literal_limit,
    }


def _signature_violations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    labels = {
        "aggregates": "聚合函数或DISTINCT口径发生变化",
        "join_count": "JOIN数量发生变化",
        "join_types": "JOIN类型发生变化",
        "join_predicate_count": "JOIN连接条件数量发生变化",
        "group_expression_count": "GROUP BY维度数量发生变化",
        "filter_clause_count": "过滤子句数量发生变化",
        "filter_predicate_count": "过滤条件数量发生变化",
        "filter_literals": "过滤常量或时间范围发生变化",
        "filter_operators": "过滤比较方式发生变化",
        "filter_boolean_operators": "AND、OR或NOT逻辑发生变化",
        "select_distinct_count": "SELECT DISTINCT口径发生变化",
        "set_operation_count": "UNION等集合运算结构发生变化",
        "order_expression_count": "排序字段数量发生变化",
        "descending_order_count": "排序方向发生变化",
    }
    violations = [
        message
        for key, message in labels.items()
        if baseline[key] != candidate[key]
    ]
    if (
        baseline["literal_limit"] is not None
        and baseline["literal_limit"] != candidate["literal_limit"]
    ):
        violations.append("业务LIMIT发生变化或被删除")
    return violations


def evaluate_sql_repair(
    *,
    input_sql: str,
    candidate_sql: str,
    original_sql: str = "",
    previous_sqls: list[str] | None = None,
    dialect: str = "mysql",
) -> SQLRepairGuardResultState:
    """Detect no-op repairs, cycles and broad business-semantic drift."""
    input_fingerprint = sql_fingerprint(input_sql, dialect)
    candidate_fingerprint = sql_fingerprint(candidate_sql, dialect)
    details: dict[str, Any] = {
        "input_fingerprint": input_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
    }

    if candidate_fingerprint == input_fingerprint:
        return SQLRepairGuardResultState(
            passed=False,
            code="NO_CHANGE",
            message="修复后的SQL与修复前相同，继续重试不会产生收益",
            fingerprint=candidate_fingerprint,
            violations=["SQL没有发生有效变化"],
            details=details,
        )

    seen_sqls = [original_sql, *(previous_sqls or [])]
    seen_fingerprints = {
        sql_fingerprint(sql, dialect)
        for sql in seen_sqls
        if (sql or "").strip()
    }
    if candidate_fingerprint in seen_fingerprints:
        return SQLRepairGuardResultState(
            passed=False,
            code="REPAIR_CYCLE",
            message="修复结果回到了之前出现过的SQL，检测到修复循环",
            fingerprint=candidate_fingerprint,
            violations=["候选SQL在修复历史中已经出现"],
            details=details,
        )

    candidate_expression = _parse_single_query(candidate_sql, dialect)
    input_expression = _parse_single_query(input_sql, dialect)
    if candidate_expression is None or input_expression is None:
        return SQLRepairGuardResultState(
            passed=True,
            code="DEFERRED_TO_AUDIT",
            message="当前SQL无法做可靠的语义对比，交由AST安全审计继续判断",
            fingerprint=candidate_fingerprint,
            violations=[],
            details=details,
        )

    candidate_signature = _semantic_signature(candidate_expression, dialect)
    baselines: list[tuple[str, exp.Query]] = [("previous", input_expression)]
    original_expression = _parse_single_query(original_sql, dialect)
    if (
        original_expression is not None
        and sql_fingerprint(original_sql, dialect) != input_fingerprint
    ):
        baselines.append(("original", original_expression))

    violations: list[str] = []
    baseline_signatures: dict[str, Any] = {}
    for name, expression in baselines:
        signature = _semantic_signature(expression, dialect)
        baseline_signatures[name] = signature
        violations.extend(_signature_violations(signature, candidate_signature))

    violations = list(dict.fromkeys(violations))
    details.update({
        "baseline_signatures": baseline_signatures,
        "candidate_signature": candidate_signature,
    })
    if violations:
        return SQLRepairGuardResultState(
            passed=False,
            code="SEMANTIC_DRIFT",
            message="修复改动超出了执行错误所需的最小范围，已终止以保护业务语义",
            fingerprint=candidate_fingerprint,
            violations=violations,
            details=details,
        )

    return SQLRepairGuardResultState(
        passed=True,
        code="OK",
        message="修复未重复、未循环，核心业务结构保持稳定",
        fingerprint=candidate_fingerprint,
        violations=[],
        details=details,
    )


def repair_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "检查SQL修复收敛性", "status": "running"})

    history = list(state.get("repair_history", []))
    current_attempt = history[-1] if history else None
    input_sql = current_attempt["input_sql"] if current_attempt else state.get("original_sql", "")
    candidate_sql = current_attempt["candidate_sql"] if current_attempt else state.get("sql", "")
    previous_sqls = [
        attempt["candidate_sql"]
        for attempt in history[:-1]
    ]
    db_info = state.get("db_info") or {}
    dialect = str(db_info.get("dialect") or "mysql").casefold()

    result = evaluate_sql_repair(
        input_sql=input_sql,
        candidate_sql=candidate_sql,
        original_sql=state.get("original_sql", input_sql),
        previous_sqls=previous_sqls,
        dialect=dialect,
    )

    if current_attempt is not None:
        current_attempt = dict(current_attempt)
        current_attempt.update({
            "input_fingerprint": sql_fingerprint(input_sql, dialect),
            "candidate_fingerprint": result["fingerprint"],
            "guard_code": result["code"],
            "guard_message": result["message"],
        })
        history[-1] = current_attempt

    writer({
        "type": "sql_repair_guard",
        "status": "passed" if result["passed"] else "stopped",
        "code": result["code"],
        "message": result["message"],
        "violations": result["violations"],
        "attempt": state.get("retry_count", 0),
    })

    if not result["passed"]:
        error = json.dumps(
            {
                "source": "sql_repair_guard",
                "code": result["code"],
                "message": result["message"],
                "violations": result["violations"],
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "检查SQL修复收敛性", "status": "error"})
        writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"SQL修复保护终止流程: {error}")
        return {
            "error": error,
            "repair_history": history,
            "repair_guard_result": result,
            "repair_stop_reason": result["code"],
        }

    writer({"type": "progress", "step": "检查SQL修复收敛性", "status": "success"})
    logger.info(f"SQL修复保护检查通过: {result['code']}")
    return {
        "error": None,
        "repair_history": history,
        "repair_guard_result": result,
        "repair_stop_reason": None,
    }
