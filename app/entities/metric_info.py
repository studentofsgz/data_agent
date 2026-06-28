from dataclasses import dataclass,field


@dataclass
class MetricInfo:
    id: str
    name: str
    description: str
    expression: str = ""
    default_aggregation: str = ""
    dimensions: list[str] = field(default_factory=list)
    time_granularity: str = ""
    relevant_columns: list[str] = field(default_factory=list)
    alias: list[str] = field(default_factory=list)
