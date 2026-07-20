import asyncio
import json
import unittest

from app.agent.graph import route_after_audit, route_after_validation
from app.agent.nodes.audit_sql import MAX_LIMIT, audit_sql, audit_sql_text
from app.agent.nodes.execute_sql import execute_sql
from app.agent.state import MAX_SQL_RETRIES


TABLE_INFOS = [
    {
        "name": "fact_order",
        "role": "fact",
        "description": "订单事实表",
        "columns": [
            {"name": "order_id"},
            {"name": "order_amount"},
            {"name": "region_id"},
        ],
    },
    {
        "name": "dim_region",
        "role": "dim",
        "description": "地区维表",
        "columns": [
            {"name": "region_id"},
            {"name": "region_name"},
        ],
    },
]


class FakeRuntime:
    def __init__(self, context=None):
        self.events = []
        self.context = context or {}

    def stream_writer(self, event):
        self.events.append(event)


class SQLAuditTests(unittest.TestCase):
    def audit(self, sql: str):
        return audit_sql_text(sql, TABLE_INFOS)

    def test_valid_join_is_normalized_and_limited(self):
        result = self.audit(
            "SELECT r.region_name, SUM(o.order_amount) AS total_sales "
            "FROM fact_order o JOIN dim_region r ON o.region_id = r.region_id "
            "GROUP BY r.region_name ORDER BY total_sales DESC"
        )

        self.assertTrue(result["passed"])
        self.assertEqual("OK", result["code"])
        self.assertEqual(MAX_LIMIT, result["limit"])
        self.assertTrue(result["limit_added"])
        self.assertIn("LIMIT 10000", result["sql"])
        self.assertEqual(["dim_region", "fact_order"], result["tables"])

    def test_existing_large_limit_is_capped(self):
        result = self.audit("SELECT order_id FROM fact_order LIMIT 50000")

        self.assertTrue(result["passed"])
        self.assertTrue(result["limit_capped"])
        self.assertEqual(MAX_LIMIT, result["limit"])
        self.assertIn("LIMIT 10000", result["sql"])

    def test_multi_statement_with_hidden_drop_is_rejected(self):
        result = self.audit(
            "SELECT order_id FROM fact_order; /* harmless-looking comment */ "
            "DROP TABLE fact_order"
        )

        self.assertFalse(result["passed"])
        self.assertEqual("MULTI_STATEMENT", result["code"])

    def test_write_statements_are_rejected_by_ast_type(self):
        for sql in (
            "DELETE FROM fact_order",
            "UPDATE fact_order SET order_amount = 0",
            "INSERT INTO fact_order (order_id) VALUES ('x')",
            "DROP TABLE fact_order",
        ):
            with self.subTest(sql=sql):
                result = self.audit(sql)
                self.assertFalse(result["passed"])
                self.assertEqual("WRITE_OR_CONTROL_OPERATION", result["code"])

    def test_unknown_table_and_column_are_rejected(self):
        table_result = self.audit("SELECT order_id FROM private_order")
        column_result = self.audit("SELECT customer_password FROM fact_order")

        self.assertEqual("UNKNOWN_TABLE", table_result["code"])
        self.assertEqual("UNKNOWN_COLUMN", column_result["code"])

    def test_table_aliases_are_checked_against_the_source_table(self):
        valid = self.audit("SELECT o.order_amount FROM fact_order AS o")
        invalid = self.audit("SELECT o.region_name FROM fact_order AS o")

        self.assertTrue(valid["passed"])
        self.assertEqual("UNKNOWN_COLUMN", invalid["code"])

    def test_projection_alias_does_not_whitelist_a_fake_source_column(self):
        invalid = self.audit("SELECT secret AS secret FROM fact_order")
        valid_order_alias = self.audit(
            "SELECT SUM(order_amount) AS total_sales FROM fact_order "
            "ORDER BY total_sales"
        )

        self.assertEqual("UNKNOWN_COLUMN", invalid["code"])
        self.assertTrue(valid_order_alias["passed"])

    def test_cte_union_and_count_star_are_allowed(self):
        cte = self.audit(
            "WITH regional AS ("
            "SELECT region_id, SUM(order_amount) AS total FROM fact_order "
            "GROUP BY region_id) "
            "SELECT region_id, total FROM regional"
        )
        union = self.audit(
            "SELECT order_id FROM fact_order "
            "UNION ALL SELECT order_id FROM fact_order"
        )
        count_star = self.audit("SELECT COUNT(*) AS order_count FROM fact_order")

        self.assertTrue(cte["passed"])
        self.assertTrue(union["passed"])
        self.assertTrue(count_star["passed"])

    def test_unknown_derived_column_is_rejected(self):
        result = self.audit(
            "SELECT regional.secret FROM ("
            "SELECT region_id, SUM(order_amount) AS total FROM fact_order "
            "GROUP BY region_id) AS regional"
        )

        self.assertEqual("UNKNOWN_COLUMN", result["code"])

    def test_select_star_and_dangerous_functions_are_rejected(self):
        wildcard = self.audit("SELECT * FROM fact_order")
        sleep = self.audit("SELECT SLEEP(10)")

        self.assertEqual("WILDCARD_NOT_ALLOWED", wildcard["code"])
        self.assertEqual("DANGEROUS_FUNCTION", sleep["code"])

    def test_cross_schema_and_dynamic_limit_are_rejected(self):
        qualified = self.audit("SELECT order_id FROM other_db.fact_order")
        dynamic_limit = self.audit("SELECT order_id FROM fact_order LIMIT ?")

        self.assertEqual("QUALIFIED_TABLE_NOT_ALLOWED", qualified["code"])
        self.assertEqual("INVALID_LIMIT", dynamic_limit["code"])

    def test_audit_node_emits_structured_final_error(self):
        runtime = FakeRuntime()
        state = {
            "sql": "SELECT * FROM fact_order",
            "table_infos": TABLE_INFOS,
            "db_info": {"dialect": "mysql", "version": "8"},
            "retry_count": MAX_SQL_RETRIES,
        }

        update = audit_sql(state, runtime)
        payload = json.loads(update["error"])

        self.assertEqual("WILDCARD_NOT_ALLOWED", payload["code"])
        self.assertTrue(any(event["type"] == "sql_audit" for event in runtime.events))
        self.assertTrue(any(event["type"] == "error" for event in runtime.events))


class SQLSafetyRoutingTests(unittest.TestCase):
    def test_audit_and_validation_errors_never_route_to_execution(self):
        retryable = {"error": "bad sql", "retry_count": 0}
        exhausted = {"error": "bad sql", "retry_count": MAX_SQL_RETRIES}
        passed = {"error": None, "retry_count": 0}

        self.assertEqual("correct_sql", route_after_audit(retryable))
        self.assertEqual("end", route_after_audit(exhausted))
        self.assertEqual("authorize_sql", route_after_audit(passed))
        self.assertEqual("correct_sql", route_after_validation(retryable))
        self.assertEqual("end", route_after_validation(exhausted))
        self.assertEqual("query_plan_guard", route_after_validation(passed))

    def test_execute_node_refuses_state_with_an_error(self):
        class Repository:
            called = False

            async def execute_sql(self, sql):
                self.called = True
                return []

        repository = Repository()
        runtime = FakeRuntime({"dw_mysql_repository": repository})

        update = asyncio.run(
            execute_sql(
                {"sql": "SELECT order_id FROM fact_order", "error": "not approved"},
                runtime,
            )
        )

        self.assertFalse(repository.called)
        self.assertEqual("not approved", update["error"])
        self.assertTrue(any(event["type"] == "error" for event in runtime.events))


if __name__ == "__main__":
    unittest.main()
