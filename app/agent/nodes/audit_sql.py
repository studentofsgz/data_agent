from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger

# 硬规则：禁止任何写操作
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "RENAME",
]

DEFAULT_LIMIT = 10000


def audit_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "安全审计SQL", "status": "running"})

    sql = state["sql"]
    sql_upper = sql.upper()

    # 1. 禁止写操作（用空格边界匹配，防止误伤）
    for kw in FORBIDDEN_KEYWORDS:
        if f" {kw} " in f" {sql_upper} ":
            raise ValueError(f"SQL包含禁止操作: {kw}")

    # 2. 只允许 SELECT
    if not sql_upper.strip().startswith("SELECT"):
        raise ValueError(f"只允许SELECT查询，当前: {sql[:50]}...")

    # 3. 自动加 LIMIT
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";").rstrip() + f" LIMIT {DEFAULT_LIMIT}"

    logger.info(f"SQL安全审计通过: {sql}")

    writer({"type": "progress", "step": "安全审计SQL", "status": "success"})
    return {"sql": sql}
