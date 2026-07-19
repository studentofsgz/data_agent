import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.entities.value_info import ValueInfo
from app.prompt.prompt_loader import load_prompt


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段取值", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    value_es_repository = runtime.context["value_es_repository"]

    try:
        # 使用LLM扩展关键词
        prompt = PromptTemplate(template=load_prompt("extend_keywords_for_value_recall"), input_variables=["query"])
        output_parser = JsonOutputParser()

        chain = prompt | llm | output_parser

        result = await chain.ainvoke({"query": query})

        # 使用扩展后的关键词召回字段取值
        values_map: dict[str, ValueInfo] = {}
        candidate_columns: dict[str, dict] = {}
        keywords = list(dict.fromkeys(str(keyword) for keyword in [*keywords, *result] if keyword))
        logger.info(f"召回字段取值扩展关键词：{keywords}")
        for keyword in keywords:
            values = await value_es_repository.search_with_scores(keyword)
            for value, score in values:
                value_id = value.id
                if value_id not in values_map:
                    values_map[value_id] = value
                trace = candidate_columns.setdefault(value.column_id, {
                    "column_id": value.column_id,
                    "matched_keywords": set(),
                    "best_score": None,
                    "match_count": 0,
                })
                trace["matched_keywords"].add(keyword)
                trace["best_score"] = max(trace["best_score"] or score, score)
                trace["match_count"] += 1

        retrieved_values = list(values_map.values())

        writer({
            "type": "schema_linking",
            "stage": "value_recall",
            "keywords": keywords,
            "candidate_columns": [
                {
                    **trace,
                    "matched_keywords": sorted(trace["matched_keywords"]),
                    "best_score": round(trace["best_score"], 6)
                    if trace["best_score"] is not None
                    else None,
                }
                for _, trace in sorted(candidate_columns.items())
            ],
        })

        writer({"type": "progress", "step": "召回字段取值", "status": "success"})
        logger.info(f"召回字段取值：{list(values_map.keys())}")

        return {'retrieved_values': retrieved_values}
    except Exception as e:
        writer({"type": "progress", "step": "召回字段取值", "status": "error"})
        logger.error(f"召回字段取值失败: {str(e)}")
        raise
