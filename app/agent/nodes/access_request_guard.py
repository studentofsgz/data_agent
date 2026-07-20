"""Fail closed on identity and explicit sensitive intent before retrieval."""

from __future__ import annotations

import json

from langgraph.runtime import Runtime

from app.agent.access_control import precheck_access_request
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


def access_request_guard(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "检查数据访问请求", "status": "running"})
    result = precheck_access_request(
        query=state.get("query") or "",
        access_context=state.get("access_context") or {},
    )
    writer({
        "type": "access_policy",
        "stage": "request",
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
        writer({"type": "progress", "step": "检查数据访问请求", "status": "error"})
        writer({"type": "error", "code": result["code"], "message": error})
        logger.warning(f"数据访问请求被拒绝: {error}")
        return {"access_policy_result": result, "error": error}

    writer({"type": "progress", "step": "检查数据访问请求", "status": "success"})
    return {"access_policy_result": result, "error": None}
