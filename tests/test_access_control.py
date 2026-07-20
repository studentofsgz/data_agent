import json
import unittest

import sqlglot
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agent.access_control import (
    apply_schema_access_policy,
    authorize_sql_text,
    precheck_access_request,
    resolve_access_context,
    validate_access_context,
)
from app.agent.context import DataAgentContext
from app.agent.graph import (
    route_after_access_policy,
    route_after_authorization,
    route_after_authorized_audit,
)
from app.agent.nodes.apply_access_policy import apply_access_policy
from app.agent.nodes.access_request_guard import access_request_guard
from app.agent.nodes.audit_sql import audit_sql_text
from app.agent.nodes.authorize_sql import authorize_sql
from app.agent.state import DataAgentState
from app.services.query_service import QueryService


TABLE_INFOS = [
    {
        "name": "fact_order",
        "role": "fact",
        "description": "订单事实表",
        "columns": [
            {"name": "order_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "订单ID", "alias": []},
            {"name": "customer_id", "type": "varchar", "role": "foreign_key", "examples": [], "description": "客户ID", "alias": []},
            {"name": "region_id", "type": "varchar", "role": "foreign_key", "examples": [], "description": "地区ID", "alias": []},
            {"name": "sales_amount", "type": "decimal", "role": "measure", "examples": [], "description": "销售额", "alias": []},
            {"name": "status", "type": "int", "role": "dimension", "examples": [], "description": "状态", "alias": []},
        ],
    },
    {
        "name": "dim_region",
        "role": "dim",
        "description": "地区维度",
        "columns": [
            {"name": "region_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "地区ID", "alias": []},
            {"name": "region_name", "type": "varchar", "role": "dimension", "examples": [], "description": "地区名称", "alias": []},
        ],
    },
    {
        "name": "dim_customer",
        "role": "dim",
        "description": "客户维度",
        "columns": [
            {"name": "customer_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "客户ID", "alias": []},
            {"name": "customer_name", "type": "varchar", "role": "dimension", "examples": [], "description": "客户姓名", "alias": []},
            {"name": "member_level", "type": "varchar", "role": "dimension", "examples": [], "description": "会员等级", "alias": []},
        ],
    },
]


class FakeRuntime:
    def __init__(self):
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


def _analyst():
    return resolve_access_context(principal_id="analyst-1", role="analyst")


def _manager(scope="华东"):
    return resolve_access_context(
        principal_id="manager-1",
        role="region_manager",
        region_scope=scope,
    )


class AccessPolicyTests(unittest.TestCase):
    def test_context_validation_rejects_unknown_role_and_missing_scope(self):
        unknown = resolve_access_context(principal_id="u1", role="root")
        missing_scope = resolve_access_context(
            principal_id="u1",
            role="region_manager",
        )

        self.assertEqual("ROLE_NOT_ALLOWED", validate_access_context(unknown)[0])
        self.assertEqual("ROW_SCOPE_REQUIRED", validate_access_context(missing_scope)[0])

    def test_schema_policy_hides_name_and_marks_id_aggregation_only(self):
        catalog, visible, result = apply_schema_access_policy(
            table_infos=TABLE_INFOS,
            query="统计客户数量",
            access_context=_analyst(),
        )
        original_customer = next(t for t in catalog if t["name"] == "dim_customer")
        visible_customer = next(t for t in visible if t["name"] == "dim_customer")
        original_names = {c["name"] for c in original_customer["columns"]}
        visible_names = {c["name"] for c in visible_customer["columns"]}
        customer_id = next(c for c in visible_customer["columns"] if c["name"] == "customer_id")

        self.assertTrue(result["passed"])
        self.assertIn("customer_name", original_names)
        self.assertNotIn("customer_name", visible_names)
        self.assertIn("仅允许COUNT聚合", customer_id["description"])

    def test_sensitive_natural_language_is_rejected_before_generation(self):
        precheck = precheck_access_request(
            query="列出客户姓名",
            access_context=_analyst(),
        )
        _, visible, result = apply_schema_access_policy(
            table_infos=TABLE_INFOS,
            query="列出客户姓名",
            access_context=_analyst(),
        )

        self.assertFalse(precheck["passed"])
        self.assertEqual("SENSITIVE_DATA_DENIED", precheck["code"])
        self.assertFalse(result["passed"])
        self.assertEqual([], visible)
        self.assertEqual("SENSITIVE_DATA_DENIED", result["code"])

    def test_admin_preserves_full_schema(self):
        _, visible, result = apply_schema_access_policy(
            table_infos=TABLE_INFOS,
            query="列出客户姓名",
            access_context=resolve_access_context(principal_id="admin-1", role="admin"),
        )

        customer = next(t for t in visible if t["name"] == "dim_customer")
        self.assertTrue(result["passed"])
        self.assertIn("customer_name", {c["name"] for c in customer["columns"]})


class SQLAuthorizationTests(unittest.TestCase):
    def test_denied_column_is_rejected_even_inside_cte(self):
        result = authorize_sql_text(
            sql="WITH c AS (SELECT customer_name FROM dim_customer) SELECT customer_name FROM c",
            access_context=_analyst(),
        )

        self.assertFalse(result["passed"])
        self.assertEqual("COLUMN_ACCESS_DENIED", result["code"])

    def test_restricted_id_requires_count_but_count_distinct_is_allowed(self):
        detail = authorize_sql_text(
            sql="SELECT customer_id FROM fact_order",
            access_context=_analyst(),
        )
        aggregate = authorize_sql_text(
            sql="SELECT COUNT(DISTINCT customer_id) AS customer_count FROM fact_order",
            access_context=_analyst(),
        )

        self.assertEqual("AGGREGATION_REQUIRED", detail["code"])
        self.assertTrue(aggregate["passed"])

    def test_unlisted_table_is_rejected(self):
        result = authorize_sql_text(
            sql="SELECT salary FROM internal_salary",
            access_context=_analyst(),
        )

        self.assertEqual("TABLE_ACCESS_DENIED", result["code"])

    def test_row_policy_preserves_where_and_is_parseable(self):
        result = authorize_sql_text(
            sql="SELECT SUM(f.sales_amount) AS gmv FROM fact_order AS f WHERE f.status = 1",
            access_context=_manager(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(1, result["row_policy_scopes"])
        self.assertIn("f.status = 1 AND EXISTS", result["sql"])
        self.assertIn("region_name = '华东'", result["sql"])
        sqlglot.parse_one(result["sql"], read="mysql")
        audit = audit_sql_text(result["sql"], TABLE_INFOS)
        self.assertTrue(audit["passed"], audit)

    def test_row_policy_preserves_alias_spelling(self):
        result = authorize_sql_text(
            sql="SELECT SUM(Orders.sales_amount) FROM fact_order AS Orders",
            access_context=_manager(),
        )

        self.assertTrue(result["passed"])
        self.assertIn("Orders.region_id", result["sql"])

    def test_union_injects_every_fact_scope(self):
        result = authorize_sql_text(
            sql="SELECT SUM(sales_amount) FROM fact_order UNION ALL SELECT SUM(sales_amount) FROM fact_order",
            access_context=_manager(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(2, result["row_policy_scopes"])
        self.assertEqual(2, result["sql"].count("EXISTS"))

    def test_scope_value_is_escaped_as_literal(self):
        result = authorize_sql_text(
            sql="SELECT SUM(sales_amount) FROM fact_order",
            access_context=_manager("华东' OR 1=1 --"),
        )

        self.assertTrue(result["passed"])
        self.assertIn("华东'' OR 1=1 --", result["sql"])
        sqlglot.parse_one(result["sql"], read="mysql")


class AccessNodeAndRoutingTests(unittest.TestCase):
    def test_nodes_emit_structured_events(self):
        runtime = FakeRuntime()
        policy_update = apply_access_policy(
            {
                "query": "统计销售额",
                "table_infos": TABLE_INFOS,
                "access_context": _manager(),
            },
            runtime,
        )
        authorization_update = authorize_sql(
            {
                "sql": "SELECT SUM(sales_amount) FROM fact_order",
                "db_info": {"dialect": "mysql"},
                "access_context": _manager(),
            },
            runtime,
        )

        self.assertTrue(policy_update["access_policy_result"]["passed"])
        self.assertTrue(authorization_update["authorization_result"]["passed"])
        self.assertIn("access_policy", [e["type"] for e in runtime.events])
        self.assertIn("sql_authorization", [e["type"] for e in runtime.events])

    def test_request_guard_stops_sensitive_intent_before_retrieval(self):
        runtime = FakeRuntime()
        update = access_request_guard(
            {
                "query": "列出客户姓名",
                "access_context": _analyst(),
            },
            runtime,
        )

        self.assertEqual(
            "SENSITIVE_DATA_DENIED",
            update["access_policy_result"]["code"],
        )
        self.assertIsNotNone(update["error"])
        event = next(e for e in runtime.events if e["type"] == "access_policy")
        self.assertEqual("request", event["stage"])

    def test_routes_fail_closed(self):
        self.assertEqual("confidence_guard", route_after_access_policy({
            "error": None,
            "access_policy_result": {"passed": True},
        }))
        self.assertEqual("end", route_after_access_policy({
            "error": "denied",
            "access_policy_result": {"passed": False},
        }))
        self.assertEqual("audit_authorized_sql", route_after_authorization({
            "error": None,
            "authorization_result": {"passed": True},
        }))
        self.assertEqual("end", route_after_authorization({
            "error": "denied",
            "authorization_result": {"passed": False},
        }))
        self.assertEqual("validate_sql", route_after_authorized_audit({"error": None}))
        self.assertEqual("end", route_after_authorized_audit({"error": "unsafe"}))


def _finish(state, _runtime):
    return {
        "conversation_turn": state.get("conversation_turn", 0) + 1,
        "last_query_intent": {"query": state.get("query") or ""},
    }


def _build_identity_graph():
    builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)
    builder.add_node("finish", _finish)
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=InMemorySaver())


async def _collect_sse(stream):
    events = []
    async for item in stream:
        events.append(json.loads(item.removeprefix("data: ").strip()))
    return events


class AccessIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_thread_cannot_switch_role(self):
        graph = _build_identity_graph()
        service = QueryService(None, None, None, None, None, None, graph)
        first = await _collect_sse(service.query(
            query="统计销售额",
            principal_id="u1",
            access_role="analyst",
        ))
        thread_id = first[0]["thread_id"]
        second = await _collect_sse(service.query(
            query="列出客户姓名",
            thread_id=thread_id,
            principal_id="u1",
            access_role="admin",
        ))

        self.assertEqual("ACCESS_CONTEXT_MISMATCH", second[0]["code"])

    async def test_region_manager_requires_scope(self):
        service = QueryService(None, None, None, None, None, None, _build_identity_graph())
        events = await _collect_sse(service.query(
            query="统计销售额",
            principal_id="m1",
            access_role="region_manager",
        ))

        self.assertEqual("ROW_SCOPE_REQUIRED", events[0]["code"])


if __name__ == "__main__":
    unittest.main()
