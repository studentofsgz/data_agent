from dataclasses import asdict

from app.entities.metric_info import MetricInfo
from app.models.metric_info_mysql import MetricInfoMySQL


class MetricInfoMapper:
    @staticmethod
    def to_entity(model: MetricInfoMySQL) -> MetricInfo:
        return MetricInfo(
            id=model.id,
            name=model.name,
            description=model.description,
            expression=model.expression or "",
            default_aggregation=model.default_aggregation or "",
            dimensions=model.dimensions or [],
            time_granularity=model.time_granularity or "",
            relevant_columns=model.relevant_columns or [],
            alias=model.alias or [],
        )

    @staticmethod
    def to_model(entity: MetricInfo):
        return MetricInfoMySQL(**asdict(entity))
