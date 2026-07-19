import re
from datetime import datetime

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import (
    ColumnInfoState,
    DataAgentState,
    DateInfoState,
    MetricSemanticItemState,
    MetricSemanticState,
    TableInfoState,
    TimeSemanticState,
)
from app.core.log import logger


TIME_INTENT_PATTERN = re.compile(
    r"(按月|每个?月|月度|上个月|本月|这个月|下个月|"
    r"\d{4}年|\d{1,2}月份?|\d{1,2}[号日]|"
    r"上旬|中旬|下旬|当天|今日|今天|昨日|昨天|"
    r"去年|今年|季度|按日|每日|按年|每年|近\d+天|最近\d+天)"
)


DIM_DATE_COLUMNS = [
    ColumnInfoState(name="date_id", type="int", role="primary_key", examples=[],
                    description="日期唯一标识，格式 yyyyMMdd。", alias=["日期ID", "日期"]),
    ColumnInfoState(name="year", type="int", role="dimension", examples=[],
                    description="年份。", alias=["年", "年份"]),
    ColumnInfoState(name="quarter", type="varchar", role="dimension", examples=[],
                    description="季度。", alias=["季度"]),
    ColumnInfoState(name="month", type="int", role="dimension", examples=[],
                    description="月份。", alias=["月", "月份"]),
    ColumnInfoState(name="day", type="int", role="dimension", examples=[],
                    description="日。", alias=["日", "天"]),
]

FACT_ORDER_DATE_COLUMN = ColumnInfoState(
    name="date_id",
    type="int",
    role="foreign_key",
    examples=[],
    description="关联时间维度的外键。",
    alias=["日期", "下单日期"],
)

FACT_ORDER_ORDER_AMOUNT_COLUMN = ColumnInfoState(
    name="order_amount",
    type="decimal",
    role="measure",
    examples=[],
    description="订单金额。",
    alias=["销售额", "订单金额", "收入"],
)

FACT_ORDER_ORDER_ID_COLUMN = ColumnInfoState(
    name="order_id",
    type="varchar",
    role="primary_key",
    examples=[],
    description="订单唯一标识。",
    alias=["订单ID"],
)

FACT_ORDER_ORDER_QUANTITY_COLUMN = ColumnInfoState(
    name="order_quantity",
    type="int",
    role="measure",
    examples=[],
    description="订单中商品的购买数量。",
    alias=["销量", "销售数量", "件数"],
)

FACT_ORDER_COLUMNS = {
    "fact_order.order_amount": FACT_ORDER_ORDER_AMOUNT_COLUMN,
    "fact_order.order_id": FACT_ORDER_ORDER_ID_COLUMN,
    "fact_order.order_quantity": FACT_ORDER_ORDER_QUANTITY_COLUMN,
    "fact_order.date_id": FACT_ORDER_DATE_COLUMN,
}

METRIC_DEFINITIONS = [
    MetricSemanticItemState(
        name="GMV",
        display_name="成交总额",
        expression="SUM(fact_order.order_amount)",
        aliases=["销售额", "销售总额", "销售金额", "销售收入", "成交额", "交易额", "订单总额", "消费总额", "总订单金额"],
        required_columns=["fact_order.order_amount"],
        dimensions=["dim_region.region_name", "dim_product.category", "dim_product.brand", "dim_date.month"],
    ),
    MetricSemanticItemState(
        name="AOV",
        display_name="客单价",
        expression="SUM(fact_order.order_amount) / COUNT(DISTINCT fact_order.order_id)",
        aliases=["客单价", "平均客单价", "平均订单金额", "每单均价", "订单均价", "每单金额", "平均交易金额"],
        required_columns=["fact_order.order_amount", "fact_order.order_id"],
        dimensions=["dim_customer.member_level", "dim_region.region_name", "dim_date.month"],
    ),
    MetricSemanticItemState(
        name="ORDER_COUNT",
        display_name="订单量",
        expression="COUNT(DISTINCT fact_order.order_id)",
        aliases=["订单量", "订单数", "订单数量", "订单笔数", "订单总数", "多少笔订单", "几笔订单", "下单量"],
        required_columns=["fact_order.order_id"],
        dimensions=["dim_region.region_name", "dim_product.category", "dim_date.month"],
    ),
    MetricSemanticItemState(
        name="SALES_QUANTITY",
        display_name="销量",
        expression="SUM(fact_order.order_quantity)",
        aliases=["销量", "销售量", "销售数量", "总销量", "月销量", "购买数量最多"],
        required_columns=["fact_order.order_quantity"],
        dimensions=["dim_product.category", "dim_product.brand", "dim_date.month"],
    ),
]


def _build_time_semantics(query: str) -> TimeSemanticState:
    required = bool(TIME_INTENT_PATTERN.search(query))
    rules = []
    reason = ""

    if required:
        reason = "用户问题包含时间粒度或时间范围表达，需要使用标准时间维表。"
        rules = [
            "涉及年、月、日、季度、上个月、本月、按月、每月、上旬、当天等时间语义时，必须使用 dim_date 表。",
            "必须通过 fact_order.date_id = dim_date.date_id 关联时间维表。",
            "按月统计优先使用 dim_date.year 和 dim_date.month 分组。",
            "按日或具体日期过滤优先使用 dim_date.year、dim_date.month、dim_date.day。",
            "禁止使用 LEFT(date_id)、SUBSTRING(date_id)、date_id DIV、CAST(date_id AS CHAR) 等方式直接截取 fact_order.date_id 表达时间。",
        ]

    return TimeSemanticState(
        required=required,
        reason=reason,
        table="dim_date",
        join_condition="fact_order.date_id = dim_date.date_id",
        required_columns=["date_id", "year", "quarter", "month", "day"] if required else [],
        rules=rules,
    )


def _matches_metric(query: str, metric: MetricSemanticItemState) -> bool:
    query_lower = query.lower()
    candidates = [metric["name"], metric["display_name"], *metric["aliases"]]
    return any(candidate.lower() in query_lower for candidate in candidates)


def _build_metric_semantics(query: str) -> MetricSemanticState:
    matched_metrics = [
        metric.copy()
        for metric in METRIC_DEFINITIONS
        if _matches_metric(query, metric)
    ]

    rules = []
    if matched_metrics:
        rules = [
            "命中的业务指标必须使用 metric_semantics 中提供的 expression，这是企业统一指标口径。",
            "当 metric_semantics 与召回指标 metric_infos 存在口径冲突时，以 metric_semantics 为准。",
            "不得将 AOV/客单价 写成 AVG(fact_order.order_amount)，必须使用 SUM(fact_order.order_amount) / COUNT(DISTINCT fact_order.order_id)。",
            "不得将订单量写成普通 COUNT(*)，必须使用 COUNT(DISTINCT fact_order.order_id)。",
        ]

    return MetricSemanticState(
        required=bool(matched_metrics),
        metrics=matched_metrics,
        rules=rules,
    )


def _ensure_column(table_info: TableInfoState, column: ColumnInfoState):
    if all(current["name"] != column["name"] for current in table_info["columns"]):
        table_info["columns"].append(column.copy())


def _ensure_fact_order_table(table_infos: list[TableInfoState]) -> TableInfoState:
    fact_order = next((table for table in table_infos if table["name"] == "fact_order"), None)
    if fact_order is None:
        fact_order = TableInfoState(
            name="fact_order",
            role="fact",
            description="订单事实表，记录订单数量和金额等核心指标。",
            columns=[],
        )
        table_infos.append(fact_order)
    return fact_order


def _ensure_time_tables(table_infos: list[TableInfoState]):
    dim_date = next((table for table in table_infos if table["name"] == "dim_date"), None)
    if dim_date is None:
        table_infos.append(TableInfoState(
            name="dim_date",
            role="dim",
            description="时间维度表，用于标准年、季度、月、日等多时间粒度分析。",
            columns=[column.copy() for column in DIM_DATE_COLUMNS],
        ))
    else:
        for column in DIM_DATE_COLUMNS:
            _ensure_column(dim_date, column)

    fact_order = _ensure_fact_order_table(table_infos)
    _ensure_column(fact_order, FACT_ORDER_DATE_COLUMN)


def _ensure_metric_columns(table_infos: list[TableInfoState], metric_semantics: MetricSemanticState):
    fact_order = _ensure_fact_order_table(table_infos)
    for metric in metric_semantics["metrics"]:
        for column_id in metric["required_columns"]:
            column = FACT_ORDER_COLUMNS.get(column_id)
            if column:
                _ensure_column(fact_order, column)


async def add_extra_context(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "添加额外上下文信息", "status": "running"})

    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    try:
        # 当前的时间信息
        today = datetime.today()
        # 日期
        date = today.strftime("%Y-%m-%d")
        # 星期
        weekday = today.strftime("%A")
        # 季度
        quarter = f"Q{(today.month - 1) // 3 + 1}"

        date_info = DateInfoState(date=date, weekday=weekday, quarter=quarter)
        time_semantics = _build_time_semantics(state["query"])
        metric_semantics = _build_metric_semantics(state["query"])
        table_infos = state.get("table_infos", [])

        if time_semantics["required"]:
            _ensure_time_tables(table_infos)
        if metric_semantics["required"]:
            _ensure_metric_columns(table_infos, metric_semantics)

        # 数据仓库环境信息
        db_info = await dw_mysql_repository.get_db_info()

        writer({"type": "progress", "step": "添加额外上下文信息", "status": "success"})
        logger.info(
            f"额外上下文信息：数据库信息-{db_info} 日期信息-{date_info} "
            f"时间语义-{time_semantics} 指标语义-{metric_semantics}"
        )
        return {
            "date_info": date_info,
            "db_info": db_info,
            "time_semantics": time_semantics,
            "metric_semantics": metric_semantics,
            "table_infos": table_infos,
        }
    except Exception as e:
        writer({"type": "progress", "step": "添加额外上下文信息", "status": "error"})
        logger.error(f"添加上下文失败:{str(e)}")
        raise
