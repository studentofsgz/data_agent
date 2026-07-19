from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.schema_catalog import lexical_metric_matches
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.metric_info import MetricInfo
from app.prompt.prompt_loader import load_prompt


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回指标", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context['embedding_client']
    metric_qdrant_repository = runtime.context['metric_qdrant_repository']
    meta_mysql_repository = runtime.context["meta_mysql_repository"]

    try:
        # 使用LLM扩展关键词
        prompt = PromptTemplate(template=load_prompt("extend_keywords_for_metric_recall"), input_variables=["query"])
        output_parser = JsonOutputParser()

        chain = prompt | llm | output_parser

        result = await chain.ainvoke({"query": query})

        # 使用扩展后的关键词召回指标信息
        retrieved_metrics_map: dict[str, MetricInfo] = {}
        candidate_trace: dict[str, dict] = {}

        keywords = list(dict.fromkeys(str(keyword) for keyword in [*keywords, *result] if keyword))
        logger.info(f"召回指标信息扩展关键词：{keywords}")
        for keyword in keywords:
            embedding = await embedding_client.aembed_query(keyword)
            payloads = await metric_qdrant_repository.search_with_scores(embedding)
            for payload, score in payloads:
                metric_id = payload.id
                if metric_id not in retrieved_metrics_map:
                    retrieved_metrics_map[metric_id] = payload
                trace = candidate_trace.setdefault(metric_id, {
                    "id": metric_id,
                    "sources": set(),
                    "matched_keywords": set(),
                    "matched_terms": set(),
                    "best_score": None,
                })
                trace["sources"].add("vector")
                trace["matched_keywords"].add(keyword)
                trace["best_score"] = max(trace["best_score"] or score, score)

        if app_config.schema_linking.exact_match_enabled:
            for match in lexical_metric_matches(query, keywords):
                metric_id = match["id"]
                if metric_id not in retrieved_metrics_map:
                    payload = await meta_mysql_repository.get_metric_info_by_id(metric_id)
                    if payload is None:
                        continue
                    retrieved_metrics_map[metric_id] = payload
                trace = candidate_trace.setdefault(metric_id, {
                    "id": metric_id,
                    "sources": set(),
                    "matched_keywords": set(),
                    "matched_terms": set(),
                    "best_score": None,
                })
                trace["sources"].add("exact_alias")
                trace["matched_terms"].update(match["matched_terms"])

        retrieved_metrics = list(retrieved_metrics_map.values())

        writer({
            "type": "schema_linking",
            "stage": "metric_recall",
            "keywords": keywords,
            "candidates": [
                {
                    **trace,
                    "sources": sorted(trace["sources"]),
                    "matched_keywords": sorted(trace["matched_keywords"]),
                    "matched_terms": sorted(trace["matched_terms"]),
                    "best_score": round(trace["best_score"], 6)
                    if trace["best_score"] is not None
                    else None,
                }
                for _, trace in sorted(candidate_trace.items())
            ],
        })

        writer({"type": "progress", "step": "召回指标", "status": "success"})
        logger.info(f"召回指标信息：{list(retrieved_metrics_map.keys())}")
        return {
            "retrieved_metrics": retrieved_metrics,
            "metric_recall_sources": {
                metric_id: sorted(trace["sources"])
                for metric_id, trace in candidate_trace.items()
            },
        }
    except Exception as e:
        writer({"type": "progress", "step": "召回指标", "status": "error"})
        logger.error(f"召回指标信息失败: {str(e)}")
        raise
