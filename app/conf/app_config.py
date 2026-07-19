from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from app.conf.config_loader import load_config


# 日志配置
@dataclass
class File:
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    enable: bool
    level: str


@dataclass
class LoggingConfig:
    file: File
    console: Console


# 数据库配置
@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class QdrantConfig:
    host: str
    port: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    host: str
    port: int
    model: str


@dataclass
class ESConfig:
    host: str
    port: int
    index_name: str


@dataclass
class RerankConfig:
    column_top_k: int
    metric_top_k: int
    similarity_threshold: float


@dataclass
class SchemaLinkingConfig:
    exact_match_enabled: bool
    exact_match_boost: float


@dataclass
class SQLExecutionConfig:
    plan_guard_enabled: bool
    reject_cartesian_joins: bool
    max_estimated_rows: int
    max_full_scan_rows: int
    max_join_tables: int
    timeout_seconds: float
    max_result_rows: int
    max_concurrent_queries: int


@dataclass
class AmbiguityConfig:
    enabled: bool
    stop_on_ambiguity: bool
    require_year_for_explicit_month: bool
    clarify_vague_metric: bool
    clarify_vague_time: bool
    clarify_vague_top_k: bool
    max_rounds: int


@dataclass
class ConversationConfig:
    persistent_checkpointer: bool
    checkpoint_path: str
    session_ttl_seconds: int
    max_history_turns: int
    result_preview_rows: int


@dataclass
class ConfidenceConfig:
    enabled: bool
    high_threshold: float
    low_threshold: float
    strong_similarity_score: float
    candidate_margin: float
    max_confirmation_attempts: int


@dataclass
class SQLCacheConfig:
    similarity_threshold: float
    collection_name: str


@dataclass
class LLMConfig:
    model_name: str
    api_key: str
    base_url: str


@dataclass
class AppConfig:
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    rerank: RerankConfig
    schema_linking: SchemaLinkingConfig
    sql_execution: SQLExecutionConfig
    ambiguity: AmbiguityConfig
    conversation: ConversationConfig
    confidence: ConfidenceConfig
    sql_cache: SQLCacheConfig
    llm: LLMConfig


config_file = Path(__file__).parents[2] / 'conf' / 'app_config.yaml'
context = OmegaConf.load(config_file)
schema = OmegaConf.structured(AppConfig)
app_config: AppConfig = OmegaConf.to_object(OmegaConf.merge(schema, context))

if __name__ == '__main__':
    print(app_config.es.host)
