import sqlparse
from sqlparse.tokens import Keyword
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger

DEFAULT_LIMIT = 10000


def audit_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "安全审计SQL", "status": "running"})

    sql = state["sql"]

    # 1. AST 解析，扫描所有 DML token，只允许 SELECT
    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception:
        raise ValueError("SQL 语法解析失败，拒绝执行")

    for token in parsed.flatten():
        if token.ttype is Keyword.DML:
            if token.value.upper() != "SELECT":
                raise ValueError(f"SQL 包含禁止的操作: {token.value}")

    # 2. 自动加 LIMIT（防止全表扫描）
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";").rstrip() + f" LIMIT {DEFAULT_LIMIT}"

    logger.info(f"SQL安全审计通过: {sql}")

    writer({"type": "progress", "step": "安全审计SQL", "status": "success"})
    return {"sql": sql}
