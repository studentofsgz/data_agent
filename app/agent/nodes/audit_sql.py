import json
from typing import Any

import sqlglot
from langgraph.runtime import Runtime
from sqlglot import exp
from sqlglot.errors import ParseError

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, SQLAuditResultState, TableInfoState, MAX_SQL_RETRIES
from app.core.log import logger


DEFAULT_LIMIT = 10000
MAX_LIMIT = 10000

WRITE_OR_CONTROL_NODE_NAMES = {
    "Alter",
    "Analyze",
    "Attach",
    "Cache",
    "Command",
    "Commit",
    "Copy",
    "Create",
    "Delete",
    "Detach",
    "Drop",
    "Execute",
    "Grant",
    "Insert",
    "Into",
    "LoadData",
    "Lock",
    "Merge",
    "Pragma",
    "Revoke",
    "Rollback",
    "Set",
    "Transaction",
    "TruncateTable",
    "Uncache",
    "Update",
    "Use",
}

DANGEROUS_FUNCTIONS = {
    "benchmark",
    "load_file",
    "sleep",
    "sys_exec",
    "sys_eval",
}


def _result(
    *,
    passed: bool,
    code: str,
    message: str,
    sql: str,
    tables: list[str] | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    limit_added: bool = False,
    limit_capped: bool = False,
    details: dict[str, Any] | None = None,
) -> SQLAuditResultState:
    return SQLAuditResultState(
        passed=passed,
        code=code,
        message=message,
        sql=sql,
        tables=tables or [],
        columns=columns or [],
        limit=limit,
        limit_added=limit_added,
        limit_capped=limit_capped,
        details=details or {},
    )


def _rejected(code: str, message: str, sql: str, **kwargs: Any) -> SQLAuditResultState:
    return _result(passed=False, code=code, message=message, sql=sql, **kwargs)


def _schema_whitelist(
    table_infos: list[TableInfoState],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    table_names: dict[str, str] = {}
    table_columns: dict[str, set[str]] = {}
    for table in table_infos:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        normalized_table = table_name.casefold()
        table_names[normalized_table] = table_name
        table_columns[normalized_table] = {
            str(column.get("name") or "").strip().casefold()
            for column in table.get("columns", [])
            if str(column.get("name") or "").strip()
        }
    return table_names, table_columns


def _forbidden_node(expression: exp.Expression) -> exp.Expression | None:
    forbidden_types = tuple(
        expression_type
        for name in WRITE_OR_CONTROL_NODE_NAMES
        if (expression_type := getattr(exp, name, None)) is not None
    )
    if not forbidden_types:
        return None
    return next(
        (node for node in expression.walk() if isinstance(node, forbidden_types)),
        None,
    )


def _function_name(function: exp.Func) -> str:
    if isinstance(function, exp.Anonymous):
        return str(function.name or "").casefold()
    return str(function.sql_name() or "").casefold()


def _projection_aliases(expression: exp.Expression) -> set[str]:
    return {
        str(alias.alias or "").casefold()
        for alias in expression.find_all(exp.Alias)
        if alias.alias
    }


def _is_projection_alias_reference(column: exp.Column) -> bool:
    current = column.parent
    allowed_contexts = (exp.Group, exp.Having, exp.Order, exp.Ordered, exp.Qualify)
    while current is not None and not isinstance(current, exp.Select):
        if isinstance(current, allowed_contexts):
            return True
        current = current.parent
    return False


def audit_sql_text(
    sql: str,
    table_infos: list[TableInfoState],
    *,
    dialect: str = "mysql",
    default_limit: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
) -> SQLAuditResultState:
    """Parse and enforce the SQL execution boundary without touching the database."""
    raw_sql = (sql or "").strip()
    if not raw_sql:
        return _rejected("EMPTY_SQL", "SQL为空，拒绝执行", raw_sql)

    try:
        statements = [
            statement
            for statement in sqlglot.parse(raw_sql, read=dialect)
            if statement is not None
        ]
    except (ParseError, ValueError) as exc:
        return _rejected(
            "PARSE_ERROR",
            "SQL无法解析，拒绝执行",
            raw_sql,
            details={"parser_error": str(exc)[:500]},
        )

    if len(statements) != 1:
        return _rejected(
            "MULTI_STATEMENT",
            "只允许执行一条SQL语句",
            raw_sql,
            details={"statement_count": len(statements)},
        )

    expression = statements[0]
    forbidden = _forbidden_node(expression)
    if forbidden is not None:
        return _rejected(
            "WRITE_OR_CONTROL_OPERATION",
            f"SQL包含禁止的操作: {type(forbidden).__name__}",
            raw_sql,
        )

    if not isinstance(expression, exp.Query):
        return _rejected(
            "NON_QUERY_STATEMENT",
            f"只允许查询语句，当前语句类型为{type(expression).__name__}",
            raw_sql,
        )

    dangerous_function = next(
        (
            _function_name(function)
            for function in expression.find_all(exp.Func)
            if _function_name(function) in DANGEROUS_FUNCTIONS
        ),
        None,
    )
    if dangerous_function:
        return _rejected(
            "DANGEROUS_FUNCTION",
            f"SQL包含禁止的函数: {dangerous_function}",
            raw_sql,
            details={"function": dangerous_function},
        )

    exposed_star = next(
        (
            star
            for star in expression.find_all(exp.Star)
            if not isinstance(star.parent, exp.Count)
        ),
        None,
    )
    if exposed_star is not None:
        return _rejected(
            "WILDCARD_NOT_ALLOWED",
            "禁止使用SELECT *，必须显式选择白名单字段",
            raw_sql,
        )

    allowed_tables, allowed_columns = _schema_whitelist(table_infos)
    derived_columns: dict[str, set[str]] = {}
    for cte in expression.find_all(exp.CTE):
        if cte.alias_or_name:
            derived_columns[str(cte.alias_or_name).casefold()] = {
                str(name).casefold() for name in cte.this.named_selects
            }
    for subquery in expression.find_all(exp.Subquery):
        if subquery.alias_or_name:
            derived_columns[str(subquery.alias_or_name).casefold()] = {
                str(name).casefold() for name in subquery.this.named_selects
            }
    cte_names = {
        str(cte.alias_or_name or "").casefold()
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }
    derived_output_columns = {
        column
        for columns in derived_columns.values()
        for column in columns
    }

    referenced_tables: set[str] = set()
    table_aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        normalized_table = str(table.name or "").casefold()
        if normalized_table in cte_names:
            continue
        if table.catalog or table.db:
            return _rejected(
                "QUALIFIED_TABLE_NOT_ALLOWED",
                "禁止跨数据库或跨Schema查询",
                raw_sql,
                details={"table": table.sql(dialect=dialect)},
            )
        if normalized_table not in allowed_tables:
            return _rejected(
                "UNKNOWN_TABLE",
                f"表不在当前Schema白名单中: {table.name}",
                raw_sql,
                details={"table": table.name},
            )
        referenced_tables.add(allowed_tables[normalized_table])
        table_aliases[normalized_table] = normalized_table
        if table.alias:
            table_aliases[str(table.alias).casefold()] = normalized_table

    projection_aliases = _projection_aliases(expression)
    all_allowed_columns = {
        column
        for columns in allowed_columns.values()
        for column in columns
    }
    referenced_columns: set[str] = set()
    for column in expression.find_all(exp.Column):
        column_name = str(column.name or "").casefold()
        qualifier = str(column.table or "").casefold()

        if qualifier in derived_columns:
            if column_name not in derived_columns[qualifier]:
                return _rejected(
                    "UNKNOWN_COLUMN",
                    f"字段不在派生表白名单中: {column.sql(dialect=dialect)}",
                    raw_sql,
                    details={"column": column.sql(dialect=dialect)},
                )
            referenced_columns.add(column.sql(dialect=dialect))
            continue

        if qualifier:
            source_table = table_aliases.get(qualifier)
            if source_table is None:
                return _rejected(
                    "UNKNOWN_TABLE_ALIAS",
                    f"字段使用了未知的表或别名: {column.table}",
                    raw_sql,
                    details={"column": column.sql(dialect=dialect)},
                )
            if column_name not in allowed_columns.get(source_table, set()):
                return _rejected(
                    "UNKNOWN_COLUMN",
                    f"字段不在当前Schema白名单中: {column.sql(dialect=dialect)}",
                    raw_sql,
                    details={"column": column.sql(dialect=dialect)},
                )
        elif (
            column_name not in all_allowed_columns
            and column_name not in derived_output_columns
            and not (
                column_name in projection_aliases
                and _is_projection_alias_reference(column)
            )
        ):
            return _rejected(
                "UNKNOWN_COLUMN",
                f"字段不在当前Schema白名单中: {column.name}",
                raw_sql,
                details={"column": column.name},
            )
        referenced_columns.add(column.sql(dialect=dialect))

    limit_added = False
    limit_capped = False
    limit_value: int
    limit_expression = expression.args.get("limit")
    if limit_expression is None:
        expression = expression.limit(default_limit)
        limit_value = default_limit
        limit_added = True
    else:
        raw_limit = limit_expression.expression
        if not isinstance(raw_limit, exp.Literal) or raw_limit.is_string:
            return _rejected(
                "INVALID_LIMIT",
                "LIMIT必须是非负整数常量",
                raw_sql,
            )
        try:
            limit_value = int(raw_limit.this)
        except (TypeError, ValueError):
            return _rejected(
                "INVALID_LIMIT",
                "LIMIT必须是非负整数常量",
                raw_sql,
            )
        if limit_value < 0:
            return _rejected(
                "INVALID_LIMIT",
                "LIMIT不能是负数",
                raw_sql,
            )
        if limit_value > max_limit:
            expression = expression.limit(max_limit)
            limit_value = max_limit
            limit_capped = True

    safe_sql = expression.sql(dialect=dialect)
    return _result(
        passed=True,
        code="OK",
        message="SQL安全审计通过",
        sql=safe_sql,
        tables=sorted(referenced_tables),
        columns=sorted(referenced_columns),
        limit=limit_value,
        limit_added=limit_added,
        limit_capped=limit_capped,
    )


def audit_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "安全审计SQL", "status": "running"})

    db_info = state.get("db_info") or {}
    dialect = str(db_info.get("dialect") or "mysql").casefold()
    result = audit_sql_text(
        state.get("sql", ""),
        state.get("schema_catalog") or state.get("table_infos", []),
        dialect=dialect,
    )

    writer({
        "type": "sql_audit",
        "status": "passed" if result["passed"] else "rejected",
        "code": result["code"],
        "message": result["message"],
        "tables": result["tables"],
        "columns": result["columns"],
        "limit": result["limit"],
        "limit_added": result["limit_added"],
        "limit_capped": result["limit_capped"],
        "details": result["details"],
    })

    if not result["passed"]:
        error = json.dumps(
            {
                "source": "sql_audit",
                "code": result["code"],
                "message": result["message"],
                "details": result["details"],
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "安全审计SQL", "status": "error"})
        if state.get("retry_count", 0) >= MAX_SQL_RETRIES:
            writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"SQL安全审计拒绝执行: {error}")
        return {"error": error, "audit_result": result}

    writer({"type": "progress", "step": "安全审计SQL", "status": "success"})
    logger.info(f"SQL安全审计通过: {result['sql']}")
    return {"sql": result["sql"], "error": None, "audit_result": result}
