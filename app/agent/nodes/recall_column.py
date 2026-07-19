from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.schema_catalog import lexical_column_matches
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.prompt.prompt_loader import load_prompt


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    column_qdrant_repository = runtime.context["column_qdrant_repository"]
    meta_mysql_repository = runtime.context["meta_mysql_repository"]

    try:
        # 使用LLM扩展关键词
        prompt = PromptTemplate(
            template=load_prompt("extend_keywords_for_column_recall"),
            input_variables=["query"],
        )
        output_parser = JsonOutputParser()

        chain = prompt | llm | output_parser

        result = await chain.ainvoke({"query": query})

        # 使用扩展后的关键词召回字段信息
        retrieved_columns_map: dict[str, ColumnInfo] = {}
        candidate_trace: dict[str, dict] = {}

        keywords = list(dict.fromkeys(str(keyword) for keyword in [*keywords, *result] if keyword))
        logger.info(f"召回字段信息扩展关键词：{keywords}")
        for keyword in keywords:
            embedding = await embedding_client.aembed_query(keyword)
            payloads = await column_qdrant_repository.search_with_scores(
                embedding
            )
            for payload, score in payloads:
                column_id = payload.id
                if column_id not in retrieved_columns_map:
                    retrieved_columns_map[column_id] = payload
                trace = candidate_trace.setdefault(column_id, {
                    "id": column_id,
                    "table": payload.table_id,
                    "column": payload.name,
                    "sources": set(),
                    "matched_keywords": set(),
                    "matched_terms": set(),
                    "best_score": None,
                })
                trace["sources"].add("vector")
                trace["matched_keywords"].add(keyword)
                trace["best_score"] = max(trace["best_score"] or score, score)

        if app_config.schema_linking.exact_match_enabled:
            for match in lexical_column_matches(query, keywords):
                column_id = match["id"]
                if column_id not in retrieved_columns_map:
                    payload = await meta_mysql_repository.get_column_info_by_id(column_id)
                    if payload is None:
                        continue
                    retrieved_columns_map[column_id] = payload
                payload = retrieved_columns_map[column_id]
                trace = candidate_trace.setdefault(column_id, {
                    "id": column_id,
                    "table": payload.table_id,
                    "column": payload.name,
                    "sources": set(),
                    "matched_keywords": set(),
                    "matched_terms": set(),
                    "best_score": None,
                })
                trace["sources"].add("exact_alias")
                trace["matched_terms"].update(match["matched_terms"])

        retrieved_columns = list(retrieved_columns_map.values())

        writer({
            "type": "schema_linking",
            "stage": "column_recall",
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

        writer({"type": "progress", "step": "召回字段", "status": "success"})
        logger.info(f"召回字段信息：{list(retrieved_columns_map.keys())}")
        return {
            "retrieved_columns": retrieved_columns,
            "column_recall_sources": {
                column_id: sorted(trace["sources"])
                for column_id, trace in candidate_trace.items()
            },
        }
    except Exception as e:
        writer({"type": "progress", "step": "召回字段", "status": "error"})
        logger.error(f"召回字段信息失败: {str(e)}")
        raise
