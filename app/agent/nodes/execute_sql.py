from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "执行SQL", "status": "running"})

    sql = state["sql"]

    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    try:
        result = await dw_mysql_repository.execute_sql(sql)

        writer({"type": "progress", "step": "执行SQL", "status": "success"})
        # 自动推荐图表类型
        chart = None
        if result and len(result) > 0:
            cols = list(result[0].keys())
            numeric_cols = [
                c for c in cols
                if all(isinstance(row.get(c), (int, float)) and row.get(c) is not None for row in result)
            ]
            text_cols = [c for c in cols if c not in numeric_cols]
            if len(text_cols) >= 1 and len(numeric_cols) >= 1:
                chart = {"type": "bar", "dimension": text_cols[0], "metric": numeric_cols[0]}
            elif len(numeric_cols) >= 2:
                chart = {"type": "line", "metrics": numeric_cols}

        writer({"type": "chart_suggestion", "chart": chart})
        writer({"type": "result", "data": result})
        logger.info(f"执行SQL结果: {result}")


    except Exception as e:
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        logger.error(f"执行SQL失败:{str(e)}")
        raise
