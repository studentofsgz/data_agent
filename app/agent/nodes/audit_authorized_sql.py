"""Re-audit policy-rewritten SQL against the role-visible schema."""

from __future__ import annotations

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.nodes.audit_sql import audit_sql
from app.agent.state import DataAgentState


def audit_authorized_sql(
    state: DataAgentState,
    runtime: Runtime[DataAgentContext],
):
    authorized_state = dict(state)
    # The first audit uses schema_catalog so a hidden field becomes a stable
    # authorization error. This second audit deliberately uses visible schema.
    authorized_state["schema_catalog"] = []
    return audit_sql(authorized_state, runtime)
