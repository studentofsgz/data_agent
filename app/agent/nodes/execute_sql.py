import asyncio
import json

from langgraph.runtime import Runtime

from app.agent.answer_grounding import sanitize_answer_rows
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger
from app.repositories.mysql.dw.dw_mysql_repository import SQLExecutionTimeoutError


_EXECUTION_SEMAPHORE = asyncio.Semaphore(
    max(1, app_config.sql_execution.max_concurrent_queries)
)


async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "执行SQL", "status": "running"})

    # 纵深防御：即使图路由被错误修改，也不能绕过前置安全检查。
    if state.get("error"):
        message = f"SQL仍存在错误，拒绝执行: {state['error']}"
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        writer({"type": "error", "code": "SQL_NOT_APPROVED", "message": message})
        logger.error(message)
        return {"error": state["error"]}

    audit_result = state.get("audit_result") or {}
    query_plan_result = state.get("query_plan_result") or {}
    if not audit_result.get("passed") or not query_plan_result.get("passed"):
        message = "SQL未同时通过AST审计和查询成本检查，拒绝执行"
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        writer({"type": "error", "code": "SQL_NOT_APPROVED", "message": message})
        logger.error(message)
        return {"error": message}

    sql = state["sql"]

    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    try:
        cfg = app_config.sql_execution
        async with _EXECUTION_SEMAPHORE:
            outcome = await dw_mysql_repository.execute_sql_sandboxed(
                sql,
                timeout_seconds=cfg.timeout_seconds,
                max_result_rows=cfg.max_result_rows,
            )
        result = outcome.rows

        execution_stats = {
            "elapsed_seconds": outcome.elapsed_seconds,
            "returned_rows": outcome.returned_rows,
            "truncated": outcome.truncated,
            "timeout_seconds": outcome.timeout_seconds,
            "max_result_rows": outcome.max_result_rows,
        }
        writer({
            "type": "sql_sandbox",
            "status": "success",
            **execution_stats,
        })

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
        answer_cfg = app_config.answer_generation
        result_summary = {
            "row_count": len(result),
            "columns": list(result[0].keys()) if result else [],
            "preview": sanitize_answer_rows(
                result,
                max_rows=max(0, app_config.conversation.result_preview_rows),
                max_cell_chars=answer_cfg.max_cell_chars,
            ),
            "truncated": outcome.truncated,
        }
        answer_rows = sanitize_answer_rows(
            result,
            max_rows=max(1, answer_cfg.max_rows),
            max_cell_chars=answer_cfg.max_cell_chars,
        )
        return {
            "error": None,
            "execution_stats": execution_stats,
            "result_summary": result_summary,
            "answer_rows": answer_rows,
        }

    except SQLExecutionTimeoutError as e:
        error = json.dumps(
            {
                "source": "sql_sandbox",
                "code": "SQL_EXECUTION_TIMEOUT",
                "message": str(e),
            },
            ensure_ascii=False,
        )
        writer({"type": "sql_sandbox", "status": "timeout"})
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        writer({"type": "error", "code": "SQL_EXECUTION_TIMEOUT", "message": error})
        logger.error(error)
        return {"error": error}
    except Exception as e:
        writer({"type": "sql_sandbox", "status": "error", "error": str(e)[:500]})
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        logger.error(f"执行SQL失败:{str(e)}")
        raise
