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


class SQLRepairAttemptState(TypedDict):
    attempt: int
    input_sql: str
    candidate_sql: str
    error: str
    input_fingerprint: str
    candidate_fingerprint: str
    guard_code: str
    guard_message: str


class SQLRepairGuardResultState(TypedDict):
    passed: bool
    code: str
    message: str
    fingerprint: str
    violations: list[str]
    details: dict[str, Any]


class SQLQueryPlanResultState(TypedDict):
    passed: bool
    code: str
    message: str
    estimated_rows: int
    query_cost: float | None
    join_table_count: int
    tables: list[dict[str, Any]]
    full_scan_tables: list[str]
    warnings: list[str]
    violations: list[str]
    details: dict[str, Any]


class SQLExecutionStatsState(TypedDict):
    elapsed_seconds: float
    returned_rows: int
    truncated: bool
    timeout_seconds: float
    max_result_rows: int


class QueryIntentState(TypedDict):
    query: str
    metrics: list[str]
    dimensions: list[str]
    time: dict[str, Any]
    filters: list[dict[str, str]]
    order: str | None
    top_k: int | None


class AmbiguityResultState(TypedDict):
    needs_clarification: bool
    action: str
    code: str
    codes: list[str]
    missing_slots: list[str]
    reasons: list[str]
    question: str


class DataAgentState(TypedDict):
    query: str  # 用户查询（可能已被上下文改写）
    messages: list[dict]  # 历史对话 [{"role":"user","content":"..."},...]
    query_intent: QueryIntentState  # 结构化查询意图
    ambiguity_result: AmbiguityResultState  # 问题完整性判定
    clarification_required: bool  # 是否需要向用户追问
    clarification_question: str  # 最小必要澄清问题
    keywords: list[str]  # 用户查询的关键字

    retrieved_columns: list[ColumnInfo]  # 召回的字段信息
    retrieved_values: list[ValueInfo]  # 召回的值信息
    retrieved_metrics: list[MetricInfo]  # 召回的指标信息
    column_recall_sources: dict[str, list[str]]  # 字段候选的向量、精确别名等召回来源
    metric_recall_sources: dict[str, list[str]]  # 指标候选的召回来源

    table_infos: list[TableInfoState]  # 表信息
    metric_infos: list[MetricInfoState]  # 指标信息

    date_info: DateInfoState  # 日期信息
    db_info: DBInfoState  # 数据库信息
    time_semantics: TimeSemanticState  # 时间语义规则
    metric_semantics: MetricSemanticState  # 指标语义规则

    sql: str  # 当前待审计、验证或执行的SQL
    original_sql: str  # 第一次生成的SQL，用作修复过程的语义基线

    audit_result: SQLAuditResultState  # AST安全审计结果
    query_plan: dict[str, Any]  # MySQL EXPLAIN FORMAT=JSON原始执行计划
    query_plan_result: SQLQueryPlanResultState  # 执行前成本策略判定
    execution_stats: SQLExecutionStatsState  # 执行沙箱耗时和结果截断信息
    repair_history: list[SQLRepairAttemptState]  # 每次修复的输入、输出、错误和判定
    repair_guard_result: SQLRepairGuardResultState  # 最近一次防漂移检查结果
    repair_stop_reason: str | None  # 重复、循环或语义漂移等提前停止原因

    error: str | None  # 安全审计或验证SQL时的结构化错误信息

    retry_count: int  # SQL修正重试次数
