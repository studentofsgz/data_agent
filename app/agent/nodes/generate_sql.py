import json
from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
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


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "生成SQL", "status": "running"})

    query = state["query"]
    table_infos = state["table_infos"]
    metric_infos = state["metric_infos"]
    date_info = state["date_info"]
    db_info = state["db_info"]

    try:
        prompt = PromptTemplate(template=load_prompt("generate_sql"),
                                input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info",
                                                 "examples"])
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
             "examples": examples_text,
             })

        writer({"type": "progress", "step": "生成SQL", "status": "success"})
        writer({"type": "sql_preview", "sql": result})
        logger.info(f"生成的SQL: {result}")
        return {"sql": result}
    except Exception as e:
        writer({"type": "progress", "step": "生成SQL", "status": "error"})
        logger.error(f"生成SQL失败: {str(e)}")
        raise
