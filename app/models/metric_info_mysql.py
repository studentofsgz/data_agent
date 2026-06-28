from sqlalchemy import String, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class MetricInfoMySQL(Base):
    __tablename__ = "metric_info"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="指标编码"
    )
    name: Mapped[str | None] = mapped_column(
        String(128),
        comment="指标名称"
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        comment="指标描述"
    )
    expression: Mapped[str | None] = mapped_column(
        Text,
        comment="指标SQL公式"
    )
    default_aggregation: Mapped[str | None] = mapped_column(
        String(64),
        comment="默认聚合方式"
    )
    dimensions: Mapped[dict | list | None] = mapped_column(
        JSON,
        comment="推荐分析维度"
    )
    time_granularity: Mapped[str | None] = mapped_column(
        String(32),
        comment="建议时间粒度"
    )
    relevant_columns: Mapped[dict | list | None] = mapped_column(
        JSON,
        comment="关联字段"
    )
    alias: Mapped[dict | list | None] = mapped_column(
        JSON,
        comment="指标别名"
    )
