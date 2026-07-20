import json

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.access_control import apply_schema_access_policy
from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "过滤表格", "status": "running"})

    query = state["query"]
    catalog, table_infos, access_result = apply_schema_access_policy(
        table_infos=state["table_infos"],
        query=query,
        access_context=state.get("access_context") or {},
    )
    writer({
        "type": "access_policy",
        "stage": "schema_linking",
        "status": "passed" if access_result["passed"] else "rejected",
        **access_result,
    })
    if not access_result["passed"]:
        error = json.dumps(
            {
                "source": "access_control",
                "code": access_result["code"],
                "message": access_result["message"],
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "过滤表格", "status": "error"})
        return {
            "schema_catalog": state.get("schema_catalog") or catalog,
            "table_infos": [],
            "access_policy_result": access_result,
            "error": error,
        }

    try:
        # 用LLM过滤表信息
        prompt = PromptTemplate(template=load_prompt("filter_table_info"), input_variables=["query", "table_infos"])
        output_parser = JsonOutputParser()

        chain = prompt | llm | output_parser

        result = await chain.ainvoke(
            {"query": query, "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False)})

        # 利用模型输出过滤table_infos
        # {
        #   'fact_order':['order_amount', 'region_id'],
        #   'dim_region':['region_id', 'region_name']
        # }
        for table_info in table_infos[:]:
            if table_info["name"] not in result:
                table_infos.remove(table_info)
            else:
                selected_columns = result[table_info["name"]]
                for column_info in table_info["columns"][:]:
                    if column_info["name"] not in selected_columns:
                        table_info["columns"].remove(column_info)

        writer({
            "type": "schema_linking",
            "stage": "table_filter",
            "tables": sorted(table_info["name"] for table_info in table_infos),
            "columns": sorted(
                f"{table_info['name']}.{column_info['name']}"
                for table_info in table_infos
                for column_info in table_info["columns"]
            ),
        })

        writer({"type": "progress", "step": "过滤表格", "status": "success"})
        logger.info(f"过滤后的表信息: {[table_info['name'] for table_info in table_infos]}")
        return {
            "schema_catalog": state.get("schema_catalog") or catalog,
            "table_infos": table_infos,
            "access_policy_result": access_result,
        }
    except Exception as e:
        writer({"type": "progress", "step": "过滤表格", "status": "error"})
        logger.error(f"过滤表失败:{str(e)}")
        raise
