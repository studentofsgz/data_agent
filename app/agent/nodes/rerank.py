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

    embedding_client = runtime.context["embedding_client"]
    cfg = app_config.rerank

    try:
        # 1. 原始 query 向量化
        query_emb = np.array(await embedding_client.aembed_query(query))

        # 2. 字段重排序
        if retrieved_columns:
            col_texts = [_build_text(col) for col in retrieved_columns]
            col_embs = np.array(await embedding_client.aembed_documents(col_texts))
            col_scores = [
                (float(_cosine_similarity(query_emb, col_embs[i])), retrieved_columns[i])
                for i in range(len(retrieved_columns))
            ]
            col_scores.sort(key=lambda x: x[0], reverse=True)
            retrieved_columns = [
                item for score, item in col_scores
                if score >= cfg.similarity_threshold
            ][:cfg.column_top_k]

        # 3. 指标重排序
        if retrieved_metrics:
            met_texts = [_build_text(met) for met in retrieved_metrics]
            met_embs = np.array(await embedding_client.aembed_documents(met_texts))
            met_scores = [
                (float(_cosine_similarity(query_emb, met_embs[i])), retrieved_metrics[i])
                for i in range(len(retrieved_metrics))
            ]
            met_scores.sort(key=lambda x: x[0], reverse=True)
            retrieved_metrics = [
                item for score, item in met_scores
                if score >= cfg.similarity_threshold
            ][:cfg.metric_top_k]

        writer({"type": "progress", "step": "重排序召回结果", "status": "success"})
        logger.info(f"Rerank 后字段: {[c.id for c in retrieved_columns]}")
        logger.info(f"Rerank 后指标: {[m.id for m in retrieved_metrics]}")

        return {
            "retrieved_columns": retrieved_columns,
            "retrieved_metrics": retrieved_metrics,
        }

    except Exception as e:
        writer({"type": "progress", "step": "重排序召回结果", "status": "error"})
        logger.error(f"Rerank 失败，降级使用原始召回结果: {str(e)}")

        # 降级：失败时原样透传，不阻塞流水线
        return {
            "retrieved_columns": retrieved_columns,
            "retrieved_metrics": retrieved_metrics,
        }
