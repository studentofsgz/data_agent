"""Deterministic RBAC, column policy and row-scope SQL authorization."""

from __future__ import annotations

import copy
import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.agent.state import ColumnInfoState, TableInfoState
from app.conf.app_config import RoleAccessConfig, app_config


PRINCIPAL_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")

FACT_REGION_COLUMN = ColumnInfoState(
    name="region_id",
    type="varchar",
    role="foreign_key",
    examples=[],
    description="关联地区维度的外键。",
    alias=["地区ID"],
)
DIM_REGION_ID_COLUMN = ColumnInfoState(
    name="region_id",
    type="varchar",
    role="primary_key",
    examples=[],
    description="地区唯一标识。",
    alias=["地区ID"],
)
DIM_REGION_NAME_COLUMN = ColumnInfoState(
    name="region_name",
    type="varchar",
    role="dimension",
    examples=[],
    description="标准地区名称。",
    alias=["地区", "区域"],
)


def resolve_access_context(
    principal_id: str | None = None,
    role: str | None = None,
    region_scope: str | None = None,
    *,
    source: str = "default",
) -> dict[str, str]:
    return {
        "principal_id": str(principal_id or "anonymous").strip(),
        "role": str(role or app_config.access_control.default_role).strip().casefold(),
        "region_scope": str(region_scope or "").strip(),
        "source": source,
    }


def validate_access_context(context: dict[str, Any]) -> tuple[str, str] | None:
    principal_id = str(context.get("principal_id") or "")
    role = str(context.get("role") or "").casefold()
    region_scope = str(context.get("region_scope") or "")
    if not PRINCIPAL_PATTERN.fullmatch(principal_id):
        return "INVALID_PRINCIPAL_ID", "principal_id格式不合法"
    policy = app_config.access_control.roles.get(role)
    if policy is None:
        return "ROLE_NOT_ALLOWED", f"未配置访问角色: {role}"
    if policy.row_policy_table and not region_scope:
        return "ROW_SCOPE_REQUIRED", f"角色{role}必须配置region_scope"
    if len(region_scope) > 64 or any(char in region_scope for char in "\r\n\0"):
        return "INVALID_ROW_SCOPE", "region_scope格式不合法"
    return None


def same_access_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("principal_id", "role", "region_scope")
    return all(str(left.get(key) or "") == str(right.get(key) or "") for key in keys)


def _role_policy(context: dict[str, Any]) -> RoleAccessConfig | None:
    return app_config.access_control.roles.get(str(context.get("role") or "").casefold())


def precheck_access_request(
    *,
    query: str,
    access_context: dict[str, Any],
) -> dict[str, Any]:
    """Reject invalid identities and explicit sensitive intent before retrieval."""
    role = str(access_context.get("role") or "").casefold()
    if not app_config.access_control.enabled:
        return {
            "passed": True,
            "code": "ACCESS_CONTROL_DISABLED",
            "message": "访问控制未启用",
            "role": role,
            "allowed_tables": ["*"],
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": [],
            "row_policy": {},
        }
    error = validate_access_context(access_context)
    policy = _role_policy(access_context)
    if error or policy is None:
        code, message = error or ("ROLE_NOT_ALLOWED", "角色未配置")
        return {
            "passed": False,
            "code": code,
            "message": message,
            "role": role,
            "allowed_tables": [],
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": [],
            "row_policy": {},
        }
    matched_terms = [term for term in policy.denied_query_terms if term in query]
    if matched_terms:
        return {
            "passed": False,
            "code": "SENSITIVE_DATA_DENIED",
            "message": "当前角色不能查询请求中的敏感数据",
            "role": role,
            "allowed_tables": list(policy.allowed_tables),
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": list(policy.aggregation_only_columns),
            "row_policy": {"matched_terms": matched_terms},
        }
    return {
        "passed": True,
        "code": "ACCESS_REQUEST_ALLOWED",
        "message": "访问主体和查询意图检查通过",
        "role": role,
        "allowed_tables": list(policy.allowed_tables),
        "removed_tables": [],
        "removed_columns": [],
        "aggregation_only_columns": list(policy.aggregation_only_columns),
        "row_policy": {},
    }


def _ensure_column(table: TableInfoState, column: ColumnInfoState) -> None:
    if all(item.get("name") != column["name"] for item in table.get("columns") or []):
        table.setdefault("columns", []).append(column.copy())


def _ensure_row_policy_schema(
    table_infos: list[TableInfoState],
    policy: RoleAccessConfig,
) -> None:
    fact_name = policy.row_policy_fact_table
    dimension_name = policy.row_policy_table
    if not fact_name or not dimension_name:
        return
    fact = next((item for item in table_infos if item.get("name") == fact_name), None)
    dimension = next(
        (item for item in table_infos if item.get("name") == dimension_name), None
    )
    if fact is not None:
        _ensure_column(fact, FACT_REGION_COLUMN)
        if dimension is None:
            dimension = TableInfoState(
                name=dimension_name,
                role="dim",
                description="地区维度表，用于行级数据权限。",
                columns=[],
            )
            table_infos.append(dimension)
        _ensure_column(dimension, DIM_REGION_ID_COLUMN)
        _ensure_column(dimension, DIM_REGION_NAME_COLUMN)
    elif dimension is not None:
        _ensure_column(dimension, DIM_REGION_ID_COLUMN)
        _ensure_column(dimension, DIM_REGION_NAME_COLUMN)


def apply_schema_access_policy(
    *,
    table_infos: list[TableInfoState],
    query: str,
    access_context: dict[str, Any],
) -> tuple[list[TableInfoState], list[TableInfoState], dict[str, Any]]:
    """Return an audit catalog, model-visible schema and structured policy result."""
    role = str(access_context.get("role") or "").casefold()
    policy = _role_policy(access_context)
    error = validate_access_context(access_context)
    if not app_config.access_control.enabled:
        catalog = copy.deepcopy(table_infos)
        return catalog, copy.deepcopy(table_infos), {
            "passed": True,
            "code": "ACCESS_CONTROL_DISABLED",
            "message": "访问控制未启用",
            "role": role,
            "allowed_tables": ["*"],
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": [],
            "row_policy": {},
        }
    if error or policy is None:
        code, message = error or ("ROLE_NOT_ALLOWED", "角色未配置")
        return copy.deepcopy(table_infos), [], {
            "passed": False,
            "code": code,
            "message": message,
            "role": role,
            "allowed_tables": [],
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": [],
            "row_policy": {},
        }

    request_result = precheck_access_request(
        query=query,
        access_context=access_context,
    )
    if not request_result["passed"]:
        matched_terms = (request_result.get("row_policy") or {}).get(
            "matched_terms",
            [],
        )
        return copy.deepcopy(table_infos), [], {
            "passed": False,
            "code": request_result["code"],
            "message": request_result["message"],
            "role": role,
            "allowed_tables": list(policy.allowed_tables),
            "removed_tables": [],
            "removed_columns": [],
            "aggregation_only_columns": list(policy.aggregation_only_columns),
            "row_policy": {"matched_terms": matched_terms},
        }

    catalog = copy.deepcopy(table_infos)
    _ensure_row_policy_schema(catalog, policy)
    visible = copy.deepcopy(catalog)
    wildcard = "*" in policy.allowed_tables
    allowed_tables = {item.casefold() for item in policy.allowed_tables}
    denied_columns = {item.casefold() for item in policy.denied_columns}
    aggregation_only = {item.casefold() for item in policy.aggregation_only_columns}
    removed_tables: list[str] = []
    removed_columns: list[str] = []
    filtered: list[TableInfoState] = []
    for table in visible:
        table_name = str(table.get("name") or "")
        if not wildcard and table_name.casefold() not in allowed_tables:
            removed_tables.append(table_name)
            continue
        columns = []
        for column in table.get("columns") or []:
            column_id = f"{table_name}.{column.get('name')}".casefold()
            if column_id in denied_columns:
                removed_columns.append(column_id)
                continue
            if column_id in aggregation_only:
                column = copy.deepcopy(column)
                description = str(column.get("description") or "")
                column["description"] = (
                    description + " 访问策略：仅允许COUNT聚合，不允许明细返回。"
                ).strip()
            columns.append(column)
        if columns:
            table["columns"] = columns
            filtered.append(table)

    if not filtered:
        return catalog, [], {
            "passed": False,
            "code": "NO_AUTHORIZED_SCHEMA",
            "message": "当前角色没有可用于本次查询的Schema",
            "role": role,
            "allowed_tables": list(policy.allowed_tables),
            "removed_tables": sorted(removed_tables),
            "removed_columns": sorted(removed_columns),
            "aggregation_only_columns": sorted(aggregation_only),
            "row_policy": {},
        }

    row_policy = {}
    if policy.row_policy_table:
        row_policy = {
            "table": policy.row_policy_table,
            "column": policy.row_policy_column,
            "scope": access_context.get("region_scope", ""),
            "fact_table": policy.row_policy_fact_table,
            "fact_key": policy.row_policy_fact_key,
            "dimension_key": policy.row_policy_dimension_key,
        }
    return catalog, filtered, {
        "passed": True,
        "code": "ACCESS_POLICY_APPLIED",
        "message": "Schema访问策略已应用",
        "role": role,
        "allowed_tables": list(policy.allowed_tables),
        "removed_tables": sorted(removed_tables),
        "removed_columns": sorted(removed_columns),
        "aggregation_only_columns": sorted(aggregation_only),
        "row_policy": row_policy,
    }


def _rejected(
    code: str,
    message: str,
    sql: str,
    role: str,
    *,
    tables: set[str] | None = None,
    denied_columns: list[str] | None = None,
    aggregation_violations: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "passed": False,
        "code": code,
        "message": message,
        "sql": sql,
        "role": role,
        "referenced_tables": sorted(tables or set()),
        "denied_columns": sorted(denied_columns or []),
        "aggregation_violations": sorted(aggregation_violations or []),
        "row_policy_applied": False,
        "row_policy_scopes": 0,
        "details": details or {},
    }


def _physical_sources(scope: Scope) -> dict[str, str]:
    return {
        str(alias).casefold(): str(source.name).casefold()
        for alias, (_, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    }


def _physical_source_aliases(scope: Scope) -> dict[str, str]:
    """Preserve alias spelling for policy SQL while normalizing table names."""
    return {
        str(alias): str(source.name).casefold()
        for alias, (_, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    }


def _column_ids(scope: Scope, column: exp.Column) -> set[str]:
    qualifier = str(column.table or "").casefold()
    column_name = str(column.name or "").casefold()
    if qualifier:
        current: Scope | None = scope
        while current is not None:
            source = _physical_sources(current).get(qualifier)
            if source:
                return {f"{source}.{column_name}"}
            current = current.parent
        return set()
    return {
        f"{table}.{column_name}"
        for table in set(_physical_sources(scope).values())
    }


def _direct_projection(scope: Scope, column: exp.Column) -> exp.Expression | None:
    if not isinstance(scope.expression, exp.Select):
        return None
    current: exp.Expression = column
    while current.parent is not None and current.parent is not scope.expression:
        current = current.parent
    if current.parent is scope.expression and current.arg_key == "expressions":
        return current
    return None


def _inside_count(column: exp.Column, projection: exp.Expression) -> bool:
    current = column.parent
    while current is not None and current is not projection.parent:
        if isinstance(current, exp.Count):
            return True
        current = current.parent
    return False


def _row_exists_predicate(
    *,
    fact_alias: str,
    policy: RoleAccessConfig,
    scope_value: str,
) -> exp.Exists:
    acl_alias = "__acl_region"
    subquery = (
        exp.select(exp.Literal.number(1))
        .from_(f"{policy.row_policy_table} AS {acl_alias}")
        .where(
            exp.and_(
                exp.EQ(
                    this=exp.column(policy.row_policy_dimension_key, acl_alias),
                    expression=exp.column(policy.row_policy_fact_key, fact_alias),
                ),
                exp.EQ(
                    this=exp.column(policy.row_policy_column, acl_alias),
                    expression=exp.Literal.string(scope_value),
                ),
            )
        )
    )
    return exp.Exists(this=subquery)


def authorize_sql_text(
    *,
    sql: str,
    access_context: dict[str, Any],
    dialect: str = "mysql",
) -> dict[str, Any]:
    """Authorize every SQL scope and inject row policy into physical fact reads."""
    role = str(access_context.get("role") or "").casefold()
    policy = _role_policy(access_context)
    error = validate_access_context(access_context)
    if not app_config.access_control.enabled:
        return {
            "passed": True,
            "code": "ACCESS_CONTROL_DISABLED",
            "message": "访问控制未启用",
            "sql": sql,
            "role": role,
            "referenced_tables": [],
            "denied_columns": [],
            "aggregation_violations": [],
            "row_policy_applied": False,
            "row_policy_scopes": 0,
            "details": {},
        }
    if error or policy is None:
        code, message = error or ("ROLE_NOT_ALLOWED", "角色未配置")
        return _rejected(code, message, sql, role)
    try:
        expression = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, ValueError) as exc:
        return _rejected(
            "AUTHORIZATION_PARSE_ERROR",
            "SQL权限审计解析失败",
            sql,
            role,
            details={"parser_error": str(exc)[:500]},
        )

    try:
        scopes = traverse_scope(expression)
    except Exception as exc:
        return _rejected(
            "AUTHORIZATION_SCOPE_ERROR",
            "SQL权限作用域解析失败",
            sql,
            role,
            details={"scope_error": str(exc)[:500]},
        )
    referenced_tables = {
        table
        for scope in scopes
        for table in _physical_sources(scope).values()
    }
    wildcard = "*" in policy.allowed_tables
    allowed_tables = {item.casefold() for item in policy.allowed_tables}
    denied_tables = sorted(
        table for table in referenced_tables
        if not wildcard and table not in allowed_tables
    )
    if denied_tables:
        return _rejected(
            "TABLE_ACCESS_DENIED",
            "SQL引用了当前角色无权访问的表",
            sql,
            role,
            tables=referenced_tables,
            details={"tables": denied_tables},
        )

    denied_policy = {item.casefold() for item in policy.denied_columns}
    aggregation_policy = {
        item.casefold() for item in policy.aggregation_only_columns
    }
    denied_hits: set[str] = set()
    aggregation_hits: set[str] = set()
    for scope in scopes:
        for column in scope.columns:
            column_ids = _column_ids(scope, column)
            denied_hits.update(column_ids & denied_policy)
            projection = _direct_projection(scope, column)
            restricted = column_ids & aggregation_policy
            if projection is not None and restricted and not _inside_count(column, projection):
                aggregation_hits.update(restricted)
    if denied_hits:
        return _rejected(
            "COLUMN_ACCESS_DENIED",
            "SQL引用了当前角色无权访问的字段",
            sql,
            role,
            tables=referenced_tables,
            denied_columns=sorted(denied_hits),
        )
    if aggregation_hits:
        return _rejected(
            "AGGREGATION_REQUIRED",
            "敏感标识字段只允许COUNT聚合，不允许明细返回",
            sql,
            role,
            tables=referenced_tables,
            aggregation_violations=sorted(aggregation_hits),
        )

    row_policy_scopes = 0
    if policy.row_policy_table:
        scope_value = str(access_context.get("region_scope") or "")
        for scope in scopes:
            if not isinstance(scope.expression, exp.Select):
                continue
            sources = _physical_source_aliases(scope)
            fact_aliases = [
                alias for alias, table in sources.items()
                if table == policy.row_policy_fact_table.casefold()
            ]
            dimension_aliases = [
                alias for alias, table in sources.items()
                if table == policy.row_policy_table.casefold()
            ]
            if fact_aliases:
                for fact_alias in fact_aliases:
                    scope.expression.where(
                        _row_exists_predicate(
                            fact_alias=fact_alias,
                            policy=policy,
                            scope_value=scope_value,
                        ),
                        copy=False,
                    )
                    row_policy_scopes += 1
            elif dimension_aliases:
                for dimension_alias in dimension_aliases:
                    scope.expression.where(
                        exp.EQ(
                            this=exp.column(
                                policy.row_policy_column, dimension_alias
                            ),
                            expression=exp.Literal.string(scope_value),
                        ),
                        copy=False,
                    )
                    row_policy_scopes += 1

    rewritten_sql = expression.sql(dialect=dialect)
    return {
        "passed": True,
        "code": "ROW_POLICY_APPLIED" if row_policy_scopes else "SQL_AUTHORIZED",
        "message": "SQL权限审计通过",
        "sql": rewritten_sql,
        "role": role,
        "referenced_tables": sorted(referenced_tables),
        "denied_columns": [],
        "aggregation_violations": [],
        "row_policy_applied": row_policy_scopes > 0,
        "row_policy_scopes": row_policy_scopes,
        "details": {
            "region_scope": access_context.get("region_scope", "")
            if row_policy_scopes
            else "",
        },
    }
