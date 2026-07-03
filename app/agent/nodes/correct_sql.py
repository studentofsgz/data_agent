import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from app.agent.state import MAX_SQL_RETRIES

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    retry_count = state.get("retry_count", 0) + 1

    if retry_count > MAX_SQL_RETRIES:
        writer({"type": "progress", "step": "校正SQL", "status": "error"})
        logger.warning(f"已达最大重试次数 ({MAX_SQL_RETRIES})，放弃修正")
        return {"retry_count": retry_count}

    writer({"type": "progress", "step": f"校正SQL (第{retry_count}次)", "status": "running"})

    sql = state["sql"]
    error = state["error"]

    query = state["query"]
    table_infos = state["table_infos"]
    metric_infos = state["metric_infos"]
    date_info = state["date_info"]
    db_info = state["db_info"]

    try:
        prompt = PromptTemplate(template=load_prompt("correct_sql"), input_variables=["query", "metric_infos"])
        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser

        result = await chain.ainvoke(
            {"query": query,
             "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
             "metric_infos": yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
             "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
             "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
             "sql": sql,
             "error": error
             })
        writer({"type": "progress", "step": f"校正SQL (第{retry_count}次)", "status": "success"})
        logger.info(f"校正后的SQL (第{retry_count}次): {result}")
        return {"sql": result, "retry_count": retry_count}
    except Exception as e:
        writer({"type": "progress", "step": f"校正SQL (第{retry_count}次)", "status": "error"})
        logger.error(f"校正SQL失败:{str(e)}")
        return {"retry_count": retry_count}
