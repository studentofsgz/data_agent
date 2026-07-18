from typing import Any, TypedDict

from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo

MAX_SQL_RETRIES = 3  # SQL 验证失败最大重试次数

class ColumnInfoState(TypedDict):
    name: str
    type: str
    role: str
    examples: list
    description: str
    alias: list[str]


class TableInfoState(TypedDict):
    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]


class MetricInfoState(TypedDict):
    name: str
    description: str
    expression: str
    default_aggregation: str
    dimensions: list[str]
    time_granularity: str
    relevant_columns: list[str]
    alias: list[str]


class DateInfoState(TypedDict):
    date: str
    weekday: str
    quarter: str


class DBInfoState(TypedDict):
    dialect: str
    version: str


class TimeSemanticState(TypedDict):
    required: bool
    reason: str
    table: str
    join_condition: str
    required_columns: list[str]
    rules: list[str]


class MetricSemanticItemState(TypedDict):
    name: str
    display_name: str
    expression: str
    aliases: list[str]
    required_columns: list[str]
    dimensions: list[str]


class MetricSemanticState(TypedDict):
    required: bool
    metrics: list[MetricSemanticItemState]
    rules: list[str]


class SQLAuditResultState(TypedDict):
    passed: bool
    code: str
    message: str
    sql: str
    tables: list[str]
    columns: list[str]
    limit: int | None
    limit_added: bool
    limit_capped: bool
    details: dict[str, Any]


class DataAgentState(TypedDict):
    query: str  # 用户查询（可能已被上下文改写）
    messages: list[dict]  # 历史对话 [{"role":"user","content":"..."},...]
    keywords: list[str]  # 用户查询的关键字

    retrieved_columns: list[ColumnInfo]  # 召回的字段信息
    retrieved_values: list[ValueInfo]  # 召回的值信息
    retrieved_metrics: list[MetricInfo]  # 召回的指标信息

    table_infos: list[TableInfoState]  # 表信息
    metric_infos: list[MetricInfoState]  # 指标信息

    date_info: DateInfoState  # 日期信息
    db_info: DBInfoState  # 数据库信息
    time_semantics: TimeSemanticState  # 时间语义规则
    metric_semantics: MetricSemanticState  # 指标语义规则

    sql: str  # 生成的SQL

    audit_result: SQLAuditResultState  # AST安全审计结果

    error: str | None  # 安全审计或验证SQL时的结构化错误信息

    retry_count: int  # SQL修正重试次数
