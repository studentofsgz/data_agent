import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


def _select_examples(keywords, top_k=2):
    """基于关键词交集选择最相关的 few-shot 示例"""
    examples_path = Path(__file__).parents[3] / "prompts" / "examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    scored = []
    for ex in examples:
        overlap = len(set(keywords) & set(ex["keywords"]))
        scored.append((overlap, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]


# ── SQL 缓存 ──

async def _ensure_cache_collection(client, collection_name):
    """确保缓存 collection 存在（懒初始化）"""
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )


async def _search_cache(client, collection_name, embedding, threshold):
    """在缓存中搜索语义相似的 query"""
    result = await client.query_points(
        collection_name=collection_name,
        query=embedding,
        score_threshold=threshold,
        limit=1,
    )
    if result.points:
        p = result.points[0]
        return {"sql": p.payload["sql"], "cached_query": p.payload["query"], "score": p.score}
    return None


async def _save_cache(client, collection_name, point_id, embedding, query, sql):
    """将新生成的 SQL 写入缓存"""
    await client.upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "query": query,
                "sql": sql,
                "timestamp": datetime.now().isoformat(),
            },
        )],
    )


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "生成SQL", "status": "running"})

    query = state["query"]
    table_infos = state["table_infos"]
    metric_infos = state["metric_infos"]
    date_info = state["date_info"]
    db_info = state["db_info"]
    time_semantics = state.get("time_semantics", {})
    metric_semantics = state.get("metric_semantics", {})

    try:
        # ====== 缓存检查 ======
        cache_cfg = app_config.sql_cache
        qdrant_client = qdrant_client_manager.client
        embedding_client = runtime.context["embedding_client"]
        should_use_cache = (
            not time_semantics.get("required", False)
            and not metric_semantics.get("required", False)
        )

        await _ensure_cache_collection(qdrant_client, cache_cfg.collection_name)
        query_emb = await embedding_client.aembed_query(query)
        cached = None
        if should_use_cache:
            cached = await _search_cache(qdrant_client, cache_cfg.collection_name, query_emb, cache_cfg.similarity_threshold)
            writer({
                "type": "sql_cache",
                "status": "hit" if cached else "miss",
            })
        else:
            logger.info("检测到语义规则，跳过SQL缓存查询，避免命中旧的时间或指标口径写法")
            writer({
                "type": "sql_cache",
                "status": "bypassed",
                "reason": "semantic_rules",
            })

        if cached:
            logger.info(f"SQL 缓存命中 (score={cached['score']:.4f}): 已缓存问题='{cached['cached_query']}'")
            writer({"type": "progress", "step": "生成SQL（缓存命中）", "status": "success"})
            writer({"type": "sql_preview", "sql": cached["sql"]})
            return {"sql": cached["sql"]}
        # ====== 缓存未命中，走 LLM 生成 ======

        prompt = PromptTemplate(template=load_prompt("generate_sql"),
                                input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info",
                                                 "time_semantics", "metric_semantics", "examples"])
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        examples = _select_examples(state.get("keywords", []))
        examples_text = "\n---\n".join(
            f"问题：{e['question']}\nSQL：{e['sql']}" for e in examples
        )

        result = await chain.ainvoke(
            {"query": query,
             "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
             "metric_infos": yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
             "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
             "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
             "time_semantics": yaml.dump(time_semantics, allow_unicode=True, sort_keys=False),
             "metric_semantics": yaml.dump(metric_semantics, allow_unicode=True, sort_keys=False),
             "examples": examples_text,
             })

        # ====== 写入缓存 ======
        await _save_cache(qdrant_client, cache_cfg.collection_name, str(uuid.uuid4()), query_emb, query, result)

        writer({"type": "progress", "step": "生成SQL", "status": "success"})
        writer({"type": "sql_preview", "sql": result})
        logger.info(f"生成的SQL（已缓存）: {result}")
        return {"sql": result}
    except Exception as e:
        writer({"type": "progress", "step": "生成SQL", "status": "error"})
        logger.error(f"生成SQL失败: {str(e)}")
        raise
