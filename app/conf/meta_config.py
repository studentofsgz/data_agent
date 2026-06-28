from dataclasses import dataclass
from typing import Optional


@dataclass
class ColumnConfig:
    name: str
    role: str
    description: str
    alias: list[str]
    sync: bool


@dataclass
class TableConfig:
    name: str
    role: str
    description: str
    columns: list[ColumnConfig]


@dataclass
class MetricConfig:
    name: str
    description: str
    expression: str = ""
    default_aggregation: str = ""
    dimensions: Optional[list[str]] = None
    time_granularity: str = ""
    relevant_columns: Optional[list[str]] = None
    alias: Optional[list[str]] = None


@dataclass
class MetaConfig:
    tables: Optional[list[TableConfig]] = None
    metrics: Optional[list[MetricConfig]] = None
