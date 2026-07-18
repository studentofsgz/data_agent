from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, MAX_SQL_RETRIES
from app.core.log import logger


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "验证SQL", "status": "running"})

    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    sql = state["sql"]

    try:
        await dw_mysql_repository.validate_sql(sql)
        writer({"type": "progress", "step": "验证SQL", "status": "success"})
        logger.info(f"SQL验证成功 (重试{state.get('retry_count', 0)}次): {sql}")
        return {"error": None}
    except Exception as e:
        error = str(e)
        writer({"type": "progress", "step": "验证SQL", "status": "error"})
        logger.error(f"SQL验证失败 (重试{state.get('retry_count', 0)}次): {sql}")
        if state.get("retry_count", 0) >= MAX_SQL_RETRIES:
            writer({"type": "error", "code": "SQL_VALIDATION_FAILED", "message": error})
        return {"error": error}
