"""Authorize model-generated SQL and inject configured row-level policies."""

from __future__ import annotations

import json

from langgraph.runtime import Runtime

from app.agent.access_control import authorize_sql_text
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


def authorize_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "授权SQL数据访问", "status": "running"})
    dialect = str((state.get("db_info") or {}).get("dialect") or "mysql").casefold()
    result = authorize_sql_text(
        sql=state.get("sql") or "",
        access_context=state.get("access_context") or {},
        dialect=dialect,
    )
    writer({
        "type": "sql_authorization",
        "status": "passed" if result["passed"] else "rejected",
        **result,
    })
    if not result["passed"]:
        error = json.dumps(
            {
                "source": "sql_authorization",
                "code": result["code"],
                "message": result["message"],
                "details": result.get("details") or {},
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "授权SQL数据访问", "status": "error"})
        writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"SQL权限审计拒绝执行: {error}")
        return {"authorization_result": result, "error": error}

    writer({"type": "progress", "step": "授权SQL数据访问", "status": "success"})
    logger.info(
        "SQL权限审计通过: "
        f"role={result['role']} row_policy_scopes={result['row_policy_scopes']}"
    )
    return {
        "sql": result["sql"],
        "authorization_result": result,
        "error": None,
    }
