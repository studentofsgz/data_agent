"""Apply the authenticated principal's access policy before SQL generation."""

from __future__ import annotations

import json

from langgraph.runtime import Runtime

from app.agent.access_control import apply_schema_access_policy
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


def apply_access_policy(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "应用数据访问策略", "status": "running"})
    if state.get("error"):
        writer({"type": "progress", "step": "应用数据访问策略", "status": "error"})
        return {
            "access_policy_result": state.get("access_policy_result") or {},
            "error": state["error"],
        }
    catalog, visible_schema, result = apply_schema_access_policy(
        table_infos=state.get("table_infos") or [],
        query=state.get("query") or "",
        access_context=state.get("access_context") or {},
    )
    writer({
        "type": "access_policy",
        "stage": "schema_final",
        "status": "passed" if result["passed"] else "rejected",
        **result,
    })
    if not result["passed"]:
        error = json.dumps(
            {
                "source": "access_control",
                "code": result["code"],
                "message": result["message"],
            },
            ensure_ascii=False,
        )
        writer({"type": "progress", "step": "应用数据访问策略", "status": "error"})
        writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"数据访问策略拒绝查询: {error}")
        return {
            "schema_catalog": state.get("schema_catalog") or catalog,
            "table_infos": visible_schema,
            "access_policy_result": result,
            "error": error,
        }

    writer({"type": "progress", "step": "应用数据访问策略", "status": "success"})
    return {
        "schema_catalog": state.get("schema_catalog") or catalog,
        "table_infos": visible_schema,
        "access_policy_result": result,
        "error": None,
    }
