import numpy as np
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger


def _cosine_similarity(a, b):
    """两个向量的余弦相似度，范围 [-1, 1]"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def _build_text(item):
    """把字段/指标拼成一条文本，格式与建 Qdrant 索引时一致：name + description + alias"""
    parts = [item.name]
    if item.description:
        parts.append(item.description)
    if item.alias:
        parts.extend(item.alias)
    return " ".join(parts)


async def rerank(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "重排序召回结果", "status": "running"})

    query = state["query"]
    retrieved_columns = state["retrieved_columns"]
    retrieved_metrics = state["retrieved_metrics"]
    column_sources = state.get("column_recall_sources", {})
    metric_sources = state.get("metric_recall_sources", {})

    embedding_client = runtime.context["embedding_client"]
    cfg = app_config.rerank

    try:
        # 1. 原始 query 向量化
        query_emb = np.array(await embedding_client.aembed_query(query))

        # 2. 字段重排序
        column_ranked = []
        if retrieved_columns:
            col_texts = [_build_text(col) for col in retrieved_columns]
            col_embs = np.array(await embedding_client.aembed_documents(col_texts))
            col_scores = []
            for index, item in enumerate(retrieved_columns):
                base_score = float(_cosine_similarity(query_emb, col_embs[index]))
                is_exact = "exact_alias" in column_sources.get(item.id, [])
                ranking_score = max(base_score, app_config.schema_linking.exact_match_boost) if is_exact else base_score
                col_scores.append((ranking_score, base_score, item))
            col_scores.sort(key=lambda item: item[0], reverse=True)
            selected_column_scores = [
                (ranking_score, base_score, item)
                for ranking_score, base_score, item in col_scores
                if base_score >= cfg.similarity_threshold
                or "exact_alias" in column_sources.get(item.id, [])
            ][:cfg.column_top_k]
            retrieved_columns = [item for _, _, item in selected_column_scores]
            column_ranked = [
                {
                    "id": item.id,
                    "table": item.table_id,
                    "score": round(ranking_score, 6),
                    "base_score": round(base_score, 6),
                    "sources": column_sources.get(item.id, []),
                }
                for ranking_score, base_score, item in selected_column_scores
            ]

        # 3. 指标重排序
        metric_ranked = []
        if retrieved_metrics:
            met_texts = [_build_text(met) for met in retrieved_metrics]
            met_embs = np.array(await embedding_client.aembed_documents(met_texts))
            met_scores = []
            for index, item in enumerate(retrieved_metrics):
                base_score = float(_cosine_similarity(query_emb, met_embs[index]))
                is_exact = "exact_alias" in metric_sources.get(item.id, [])
                ranking_score = max(base_score, app_config.schema_linking.exact_match_boost) if is_exact else base_score
                met_scores.append((ranking_score, base_score, item))
            met_scores.sort(key=lambda item: item[0], reverse=True)
            selected_metric_scores = [
                (ranking_score, base_score, item)
                for ranking_score, base_score, item in met_scores
                if base_score >= cfg.similarity_threshold
                or "exact_alias" in metric_sources.get(item.id, [])
            ][:cfg.metric_top_k]
            retrieved_metrics = [item for _, _, item in selected_metric_scores]
            metric_ranked = [
                {
                    "id": item.id,
                    "score": round(ranking_score, 6),
                    "base_score": round(base_score, 6),
                    "sources": metric_sources.get(item.id, []),
                }
                for ranking_score, base_score, item in selected_metric_scores
            ]

        writer({
            "type": "schema_linking",
            "stage": "rerank",
            "column_top_k": cfg.column_top_k,
            "metric_top_k": cfg.metric_top_k,
            "columns": column_ranked,
            "metrics": metric_ranked,
        })

        writer({"type": "progress", "step": "重排序召回结果", "status": "success"})
        logger.info(f"Rerank 后字段: {[c.id for c in retrieved_columns]}")
        logger.info(f"Rerank 后指标: {[m.id for m in retrieved_metrics]}")

        return {
            "retrieved_columns": retrieved_columns,
            "retrieved_metrics": retrieved_metrics,
            "column_candidate_scores": {
                item["id"]: item["score"] for item in column_ranked
            },
            "metric_candidate_scores": {
                item["id"]: item["score"] for item in metric_ranked
            },
            "schema_linking_degraded": False,
        }

    except Exception as e:
        writer({"type": "progress", "step": "重排序召回结果", "status": "error"})
        logger.error(f"Rerank 失败，降级使用原始召回结果: {str(e)}")

        writer({
            "type": "schema_linking",
            "stage": "rerank",
            "status": "degraded",
            "columns": [{"id": item.id, "table": item.table_id, "score": None} for item in retrieved_columns],
            "metrics": [{"id": item.id, "score": None} for item in retrieved_metrics],
        })

        # 降级：失败时原样透传，不阻塞流水线
        return {
            "retrieved_columns": retrieved_columns,
            "retrieved_metrics": retrieved_metrics,
            "column_candidate_scores": {
                item.id: None for item in retrieved_columns
            },
            "metric_candidate_scores": {
                item.id: None for item in retrieved_metrics
            },
            "schema_linking_degraded": True,
        }
