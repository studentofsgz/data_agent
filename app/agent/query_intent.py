"""Deterministic query-intent extraction and ambiguity policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AmbiguityPolicy:
    require_year_for_explicit_month: bool = True
    clarify_vague_metric: bool = True
    clarify_vague_time: bool = True
    clarify_vague_top_k: bool = True


METRIC_ALIASES = {
    "GMV": (
        "GMV", "销售总额", "总销售额", "销售金额", "销售收入", "销售额",
        "成交额", "交易额", "订单总额", "订单总金额", "消费总额", "总订单金额",
    ),
    "AOV": (
        "AOV", "客单价", "平均客单价", "平均订单金额", "每单均价", "订单均价",
    ),
    "ORDER_COUNT": (
        "订单量", "订单数", "订单数量", "订单笔数", "订单总数", "多少笔订单",
        "几笔订单", "下单量",
    ),
    "SALES_QUANTITY": (
        "总销量", "月销量", "销售数量", "销售量", "销量", "购买数量最多",
    ),
    "AVG_PURCHASE_QUANTITY": (
        "平均购买数量", "平均购买量", "平均每单购买数量", "平均每单购买量",
    ),
    "ORDER_AMOUNT": (
        "订单金额最高", "订单金额最低", "每笔订单金额", "单笔订单金额",
    ),
}

DIMENSION_PATTERNS = (
    ("region", re.compile(r"地区|大区|区域")),
    ("province", re.compile(r"省份|各省|省级")),
    ("city", re.compile(r"城市|各市|市级")),
    ("product", re.compile(r"商品|产品")),
    ("category", re.compile(r"品类|类别|类目")),
    ("brand", re.compile(r"品牌")),
    ("customer", re.compile(r"客户|用户")),
    ("member_level", re.compile(r"会员等级|会员级别|会员")),
    ("gender", re.compile(r"性别|男性|女性|男客户|女客户")),
)

RELATIVE_TIME_PATTERNS = (
    ("last_month", re.compile(r"上个月")),
    ("this_month", re.compile(r"本月|这个月")),
    ("next_month", re.compile(r"下个月")),
    ("last_year", re.compile(r"去年")),
    ("this_year", re.compile(r"今年")),
    ("next_year", re.compile(r"明年")),
    ("yesterday", re.compile(r"昨天|昨日")),
    ("today", re.compile(r"今天|今日|当天")),
)

EXPLICIT_YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})年")
EXPLICIT_MONTH_PATTERN = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])月份?")
EXPLICIT_DAY_PATTERN = re.compile(r"(?<!\d)(3[01]|[12]\d|0?[1-9])[日号]")
BOUNDED_RECENT_PATTERN = re.compile(r"(?:最近|近)\s*(\d+)\s*(天|日|周|个月|月|年)")
VAGUE_RECENT_PATTERN = re.compile(r"最近(?!\s*\d)|近期|近来")
TOP_K_PATTERN = re.compile(r"(?:前|TOP\s*)(\d+)", re.IGNORECASE)
EXTREME_COUNT_PATTERN = re.compile(r"(?:最高|最低|最多|最少)的?(\d+)个?")
VAGUE_TOP_K_PATTERN = re.compile(r"前几|排名靠前|排在前面|头部")
CONTEXT_REFERENCE_PATTERN = re.compile(r"这个指标|那个指标|这个结果|那个结果|上述|刚才的|它们?")
METRIC_REQUIRED_PATTERN = re.compile(r"统计|分析|汇总|表现|情况|趋势|排名|最高|最低|最多|最少|是多少|多少")


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _extract_metrics(query: str) -> list[str]:
    query_lower = query.casefold()
    return [
        metric
        for metric, aliases in METRIC_ALIASES.items()
        if any(alias.casefold() in query_lower for alias in aliases)
    ]


def _extract_dimensions(query: str) -> list[str]:
    dimensions = [
        dimension
        for dimension, pattern in DIMENSION_PATTERNS
        if pattern.search(query)
    ]
    return _deduplicate(dimensions)


def _extract_filters(query: str) -> list[dict[str, str]]:
    filters: list[dict[str, str]] = []
    for region in ("华东", "华南", "华北", "华中", "西南", "西北", "东北"):
        if region in query:
            filters.append({"field": "region", "value": region})
    for gender, normalized in (("女性", "female"), ("男性", "male")):
        if gender in query:
            filters.append({"field": "gender", "value": normalized})
    for level in ("普通会员", "白银会员", "黄金会员", "铂金会员", "钻石会员"):
        if level in query:
            filters.append({"field": "member_level", "value": level})
    return filters


def extract_query_intent(query: str) -> dict[str, Any]:
    normalized_query = str(query or "").strip()
    year_match = EXPLICIT_YEAR_PATTERN.search(normalized_query)
    month_match = EXPLICIT_MONTH_PATTERN.search(normalized_query)
    day_match = EXPLICIT_DAY_PATTERN.search(normalized_query)
    recent_match = BOUNDED_RECENT_PATTERN.search(normalized_query)

    relative_time = next(
        (
            name
            for name, pattern in RELATIVE_TIME_PATTERNS
            if pattern.search(normalized_query)
        ),
        None,
    )
    if recent_match:
        relative_time = f"recent_{recent_match.group(1)}_{recent_match.group(2)}"

    if re.search(r"每天|每日|按日|逐日", normalized_query):
        grain = "day"
    elif re.search(r"每个月|每月|按月|月度|逐月", normalized_query):
        grain = "month"
    elif re.search(r"每季度|按季度|季度", normalized_query):
        grain = "quarter"
    elif re.search(r"每年|按年|年度|逐年", normalized_query):
        grain = "year"
    elif day_match:
        grain = "day"
    elif month_match:
        grain = "month"
    elif year_match:
        grain = "year"
    else:
        grain = None

    top_match = TOP_K_PATTERN.search(normalized_query)
    if top_match:
        top_k = int(top_match.group(1))
    elif extreme_count_match := EXTREME_COUNT_PATTERN.search(normalized_query):
        top_k = int(extreme_count_match.group(1))
    elif re.search(r"最高|最低|最多|最少|第一名", normalized_query):
        top_k = 1
    else:
        top_k = None

    if re.search(r"最低|最少|升序|从低到高", normalized_query):
        order = "ASC"
    elif re.search(r"最高|最多|降序|从高到低|排名", normalized_query):
        order = "DESC"
    else:
        order = None

    return {
        "query": normalized_query,
        "metrics": _extract_metrics(normalized_query),
        "dimensions": _extract_dimensions(normalized_query),
        "time": {
            "year": int(year_match.group(1)) if year_match else None,
            "month": int(month_match.group(1)) if month_match else None,
            "day": int(day_match.group(1)) if day_match else None,
            "relative": relative_time,
            "grain": grain,
        },
        "filters": _extract_filters(normalized_query),
        "order": order,
        "top_k": top_k,
    }


def evaluate_ambiguity(
    intent: dict[str, Any],
    *,
    has_history: bool = False,
    policy: AmbiguityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or AmbiguityPolicy()
    query = str(intent.get("query") or "")
    time_info = intent.get("time") or {}
    missing_slots: list[str] = []
    reasons: list[str] = []
    codes: list[str] = []

    relative_year = time_info.get("relative") in {
        "last_year", "this_year", "next_year",
    }
    if (
        policy.require_year_for_explicit_month
        and time_info.get("month") is not None
        and time_info.get("year") is None
        and not relative_year
    ):
        missing_slots.append("time.year")
        reasons.append("问题指定了月份，但没有说明年份")
        codes.append("MISSING_YEAR_FOR_MONTH")

    if (
        time_info.get("day") is not None
        and time_info.get("month") is None
        and time_info.get("relative") not in {"today", "yesterday"}
    ):
        missing_slots.extend(["time.year", "time.month"])
        reasons.append("问题指定了日期，但没有说明月份和年份")
        codes.append("MISSING_MONTH_FOR_DAY")

    if policy.clarify_vague_time and VAGUE_RECENT_PATTERN.search(query):
        missing_slots.append("time.range")
        reasons.append("最近或近期没有明确的时间长度")
        codes.append("VAGUE_TIME_RANGE")

    if (
        policy.clarify_vague_metric
        and not intent.get("metrics")
        and METRIC_REQUIRED_PATTERN.search(query)
        and not re.search(r"有哪些|明细|名单|列表", query)
    ):
        missing_slots.append("metric")
        reasons.append("问题没有明确要分析的业务指标")
        codes.append("MISSING_METRIC")

    if (
        policy.clarify_vague_top_k
        and VAGUE_TOP_K_PATTERN.search(query)
        and intent.get("top_k") is None
    ):
        missing_slots.append("top_k")
        reasons.append("排名问题没有说明需要返回多少名")
        codes.append("MISSING_TOP_K")

    if CONTEXT_REFERENCE_PATTERN.search(query) and not has_history:
        missing_slots.append("context.reference")
        reasons.append("问题引用了上文，但当前请求没有可用历史上下文")
        codes.append("MISSING_CONTEXT")

    missing_slots = _deduplicate(missing_slots)
    codes = _deduplicate(codes)
    needs_clarification = bool(codes)
    primary_code = (
        codes[0]
        if len(codes) == 1
        else "MULTIPLE_AMBIGUITIES"
        if codes
        else "CLEAR"
    )

    asked_slot: str | None
    if "time.year" in missing_slots and time_info.get("month") is not None:
        asked_slot = "time.year"
        question = f"你指的是哪一年的{time_info['month']}月？例如2025年。"
    elif "time.month" in missing_slots:
        asked_slot = "time.month"
        question = "你指的是哪一年、哪一个月的这个日期？"
    elif "metric" in missing_slots:
        asked_slot = "metric"
        question = "你希望分析哪个指标？例如销售额、销量、订单量或客单价。"
    elif "time.range" in missing_slots:
        asked_slot = "time.range"
        question = "你说的最近是多长时间？例如最近7天、30天或3个月。"
    elif "top_k" in missing_slots:
        asked_slot = "top_k"
        question = "你希望返回排名前多少条？例如前5名或前10名。"
    elif "context.reference" in missing_slots:
        asked_slot = "context.reference"
        question = "你指的是哪个指标或哪一次查询结果？请补充具体内容。"
    else:
        asked_slot = None
        question = ""

    return {
        "needs_clarification": needs_clarification,
        "code": primary_code,
        "codes": codes,
        "missing_slots": missing_slots,
        "reasons": reasons,
        "question": question,
        "asked_slot": asked_slot,
    }


def merge_clarification_answer(
    query: str,
    *,
    asked_slot: str | None,
    answer: str,
) -> str:
    """Merge one human answer into the natural-language query deterministically."""
    original_query = str(query or "").strip()
    normalized_answer = str(answer or "").strip()
    if not normalized_answer:
        return original_query

    if asked_slot == "time.year":
        year_match = re.search(r"((?:19|20)\d{2})", normalized_answer)
        if year_match and not EXPLICIT_YEAR_PATTERN.search(original_query):
            return f"{year_match.group(1)}年{original_query}"

    if asked_slot == "time.month":
        return f"{normalized_answer}，{original_query}"

    if asked_slot == "metric":
        return f"{original_query}，要分析的指标是{normalized_answer}"

    if asked_slot == "time.range":
        return VAGUE_RECENT_PATTERN.sub(normalized_answer, original_query, count=1)

    if asked_slot == "top_k":
        count_match = re.search(r"(\d+)", normalized_answer)
        if count_match:
            replacement = f"排名前{count_match.group(1)}"
            return VAGUE_TOP_K_PATTERN.sub(replacement, original_query, count=1)

    if asked_slot == "context.reference":
        return CONTEXT_REFERENCE_PATTERN.sub(normalized_answer, original_query, count=1)

    return f"{original_query}，用户补充：{normalized_answer}"


def analyze_query_intent(
    query: str,
    *,
    has_history: bool = False,
    policy: AmbiguityPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    intent = extract_query_intent(query)
    return intent, evaluate_ambiguity(
        intent,
        has_history=has_history,
        policy=policy,
    )
