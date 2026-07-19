"""Reject executable but unacceptably expensive SQL before execution."""

from __future__ import annotations

import json

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.query_plan import QueryPlanPolicy, evaluate_query_plan
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.log import logger


async def query_plan_guard(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "检查查询成本", "status": "running"})

    cfg = app_config.sql_execution
    db_info = state.get("db_info") or {}
    result = evaluate_query_plan(
        sql=state.get("sql", ""),
        plan=state.get("query_plan") or {},
        policy=QueryPlanPolicy(
            max_estimated_rows=cfg.max_estimated_rows,
            max_full_scan_rows=cfg.max_full_scan_rows,
            max_join_tables=cfg.max_join_tables,
            reject_cartesian_joins=cfg.reject_cartesian_joins,
        ),
        dialect=str(db_info.get("dialect") or "mysql").casefold(),
    )

    if not cfg.plan_guard_enabled:
        result = {
            **result,
            "passed": True,
            "code": "PLAN_GUARD_DISABLED",
            "message": "查询成本检查已关闭",
            "violations": [],
        }

    writer({
        "type": "query_plan_guard",
        "status": "passed" if result["passed"] else "rejected",
        "code": result["code"],
        "message": result["message"],
        "estimated_rows": result["estimated_rows"],
        "query_cost": result["query_cost"],
        "join_table_count": result["join_table_count"],
        "full_scan_tables": result["full_scan_tables"],
        "warnings": result["warnings"],
        "violations": result["violations"],
        "tables": result["tables"],
    })

    if not result["passed"]:
        # EXPLAIN opened an implicit transaction; release it before ending the graph.
        repository = runtime.context.get("dw_mysql_repository")
        if repository is not None and hasattr(repository, "rollback"):
            await repository.rollback()
        error = json.dumps(
            {
                "source": "query_plan_guard",
                "code": result["code"],
                "message": result["message"],
                "details": result["details"],
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "检查查询成本", "status": "error"})
        writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"查询成本策略拒绝执行: {error}")
        return {"error": error, "query_plan_result": result}

    writer({"type": "progress", "step": "检查查询成本", "status": "success"})
    logger.info(
        f"查询成本检查通过: estimated_rows={result['estimated_rows']} "
        f"query_cost={result['query_cost']}"
    )
    return {"error": None, "query_plan_result": result}
