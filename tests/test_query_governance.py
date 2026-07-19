import asyncio
import json
import unittest

from app.agent.graph import (
    route_after_execution,
    route_after_query_plan,
    route_after_validation,
)
from app.agent.nodes.execute_sql import execute_sql
from app.agent.nodes.query_plan_guard import query_plan_guard
from app.agent.nodes.validate_sql import validate_sql
from app.agent.query_plan import QueryPlanPolicy, evaluate_query_plan
from app.repositories.mysql.dw.dw_mysql_repository import (
    SQLExecutionOutcome,
    SQLExecutionTimeoutError,
)


SMALL_INDEXED_PLAN = {
    "query_block": {
        "cost_info": {"query_cost": "12.50"},
        "ordering_operation": {
            "using_filesort": True,
            "nested_loop": [
                {
                    "table": {
                        "table_name": "r",
                        "access_type": "ALL",
                        "rows_examined_per_scan": 4,
                        "rows_produced_per_join": 4,
                    }
                },
                {
                    "table": {
                        "table_name": "o",
                        "access_type": "ref",
                        "key": "idx_region_id",
                        "possible_keys": ["idx_region_id"],
                        "rows_examined_per_scan": 25,
                        "rows_produced_per_join": 100,
                    }
                },
            ],
        },
    }
}


class FakeRuntime:
    def __init__(self, context=None):
        self.context = context or {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


class QueryPlanPolicyTests(unittest.TestCase):
    def policy(self, **overrides):
        values = {
            "max_estimated_rows": 1_000_000,
            "max_full_scan_rows": 100_000,
            "max_join_tables": 8,
            "reject_cartesian_joins": True,
        }
        values.update(overrides)
        return QueryPlanPolicy(**values)

    def test_small_indexed_plan_passes_and_keeps_warnings(self):
        result = evaluate_query_plan(
            sql=(
                "SELECT r.region_name, SUM(o.order_amount) FROM fact_order o "
                "JOIN dim_region r ON o.region_id = r.region_id "
                "GROUP BY r.region_name"
            ),
            plan=SMALL_INDEXED_PLAN,
            policy=self.policy(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual("PLAN_OK", result["code"])
        self.assertEqual(100, result["estimated_rows"])
        self.assertEqual(12.5, result["query_cost"])
        self.assertEqual(["USING_FILESORT"], result["warnings"])

    def test_cartesian_join_is_rejected_even_when_plan_is_small(self):
        result = evaluate_query_plan(
            sql=(
                "SELECT o.order_id, r.region_name FROM fact_order o "
                "CROSS JOIN dim_region r"
            ),
            plan=SMALL_INDEXED_PLAN,
            policy=self.policy(),
        )

        self.assertFalse(result["passed"])
        self.assertEqual("CARTESIAN_JOIN", result["code"])

    def test_large_full_scan_and_excessive_estimate_are_rejected(self):
        plan = {
            "query_block": {
                "table": {
                    "table_name": "fact_order",
                    "access_type": "ALL",
                    "rows_examined_per_scan": 2_000_000,
                    "rows_produced_per_join": 2_000_000,
                }
            }
        }
        result = evaluate_query_plan(
            sql="SELECT order_id FROM fact_order",
            plan=plan,
            policy=self.policy(),
        )

        self.assertEqual("LARGE_FULL_SCAN", result["code"])
        self.assertEqual(["fact_order"], result["full_scan_tables"])
        self.assertIn("ESTIMATED_ROWS_LIMIT_EXCEEDED", result["violations"])

    def test_invalid_explain_payload_fails_closed(self):
        result = evaluate_query_plan(
            sql="SELECT order_id FROM fact_order",
            plan="not-json",
            policy=self.policy(),
        )

        self.assertFalse(result["passed"])
        self.assertEqual("INVALID_QUERY_PLAN", result["code"])


class QueryGovernanceNodeTests(unittest.TestCase):
    def test_only_successful_execution_is_saved_to_conversation_memory(self):
        self.assertEqual("remember_turn", route_after_execution({"error": None}))
        self.assertEqual("end", route_after_execution({"error": "timeout"}))

    def test_validation_returns_explain_plan(self):
        class Repository:
            async def explain_sql(self, sql):
                self.sql = sql
                return SMALL_INDEXED_PLAN

        repository = Repository()
        runtime = FakeRuntime({"dw_mysql_repository": repository})
        update = asyncio.run(
            validate_sql(
                {"sql": "SELECT order_id FROM fact_order", "retry_count": 0},
                runtime,
            )
        )

        self.assertIs(update["query_plan"], SMALL_INDEXED_PLAN)
        self.assertIsNone(update["error"])

    def test_cost_rejection_ends_instead_of_triggering_llm_repair(self):
        plan = {
            "query_block": {
                "table": {
                    "table_name": "fact_order",
                    "access_type": "ALL",
                    "rows_examined_per_scan": 2_000_000,
                    "rows_produced_per_join": 2_000_000,
                }
            }
        }
        runtime = FakeRuntime()
        update = asyncio.run(
            query_plan_guard(
                {
                    "sql": "SELECT order_id FROM fact_order",
                    "query_plan": plan,
                    "db_info": {"dialect": "mysql"},
                },
                runtime,
            )
        )

        self.assertEqual("LARGE_FULL_SCAN", json.loads(update["error"])["code"])
        self.assertEqual("end", route_after_query_plan(update))
        self.assertTrue(any(e["type"] == "query_plan_guard" for e in runtime.events))

    def test_routing_requires_cost_guard_before_execution(self):
        self.assertEqual(
            "query_plan_guard",
            route_after_validation({"error": None}),
        )
        self.assertEqual(
            "execute_sql",
            route_after_query_plan({
                "error": None,
                "query_plan_result": {"passed": True},
            }),
        )
        self.assertEqual(
            "end",
            route_after_query_plan({
                "error": "too expensive",
                "query_plan_result": {"passed": False},
            }),
        )

    def test_execution_sandbox_emits_row_and_truncation_stats(self):
        class Repository:
            async def execute_sql_sandboxed(self, sql, **kwargs):
                return SQLExecutionOutcome(
                    rows=[{"order_id": "o1"}],
                    elapsed_seconds=0.02,
                    returned_rows=1,
                    truncated=False,
                    timeout_seconds=kwargs["timeout_seconds"],
                    max_result_rows=kwargs["max_result_rows"],
                )

        runtime = FakeRuntime({"dw_mysql_repository": Repository()})
        update = asyncio.run(
            execute_sql(
                {
                    "sql": "SELECT order_id FROM fact_order LIMIT 1",
                    "error": None,
                    "audit_result": {"passed": True},
                    "query_plan_result": {"passed": True},
                },
                runtime,
            )
        )

        self.assertEqual(1, update["execution_stats"]["returned_rows"])
        self.assertTrue(any(e["type"] == "sql_sandbox" for e in runtime.events))
        self.assertTrue(any(e["type"] == "result" for e in runtime.events))

    def test_execution_timeout_becomes_structured_error(self):
        class Repository:
            async def execute_sql_sandboxed(self, sql, **kwargs):
                raise SQLExecutionTimeoutError("timeout")

        runtime = FakeRuntime({"dw_mysql_repository": Repository()})
        update = asyncio.run(
            execute_sql(
                {
                    "sql": "SELECT order_id FROM fact_order",
                    "error": None,
                    "audit_result": {"passed": True},
                    "query_plan_result": {"passed": True},
                },
                runtime,
            )
        )

        self.assertEqual("SQL_EXECUTION_TIMEOUT", json.loads(update["error"])["code"])
        self.assertTrue(any(e.get("status") == "timeout" for e in runtime.events))


if __name__ == "__main__":
    unittest.main()
