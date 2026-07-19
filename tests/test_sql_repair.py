import json
import unittest

from app.agent.graph import route_after_repair_guard
from app.agent.nodes.add_extra_context import _ensure_time_tables
from app.agent.nodes.repair_guard import evaluate_sql_repair, repair_guard


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.context = {}

    def stream_writer(self, event):
        self.events.append(event)


class SQLRepairGuardTests(unittest.TestCase):
    def evaluate(self, input_sql, candidate_sql, **kwargs):
        return evaluate_sql_repair(
            input_sql=input_sql,
            candidate_sql=candidate_sql,
            original_sql=kwargs.pop("original_sql", input_sql),
            **kwargs,
        )

    def test_format_only_or_identical_repair_is_stopped(self):
        result = self.evaluate(
            "SELECT order_id FROM fact_order",
            " select ORDER_ID from FACT_ORDER; ",
        )

        self.assertFalse(result["passed"])
        self.assertEqual("NO_CHANGE", result["code"])

    def test_repair_cycle_is_stopped(self):
        result = self.evaluate(
            "SELECT order_no FROM fact_order",
            "SELECT order_id FROM fact_order",
            original_sql="SELECT order_id FROM fact_order",
            previous_sqls=["SELECT order_no FROM fact_order"],
        )

        self.assertFalse(result["passed"])
        self.assertEqual("REPAIR_CYCLE", result["code"])

    def test_small_identifier_correction_is_allowed(self):
        result = self.evaluate(
            "SELECT r.region_name, SUM(o.amount) FROM fact_order o "
            "JOIN dim_region r ON o.region_id = r.region_id "
            "WHERE o.status = 'paid' GROUP BY r.region_name",
            "SELECT r.region_name, SUM(o.order_amount) FROM fact_order o "
            "JOIN dim_region r ON o.region_id = r.region_id "
            "WHERE o.order_status = 'paid' GROUP BY r.region_name",
        )

        self.assertTrue(result["passed"])
        self.assertEqual("OK", result["code"])

    def test_removing_filter_or_changing_aggregate_is_semantic_drift(self):
        removed_filter = self.evaluate(
            "SELECT region_id, SUM(order_amount) FROM fact_order "
            "WHERE order_status = 'paid' GROUP BY region_id",
            "SELECT region_id, SUM(order_amount) FROM fact_order GROUP BY region_id",
        )
        changed_metric = self.evaluate(
            "SELECT region_id, SUM(order_amount) FROM fact_order GROUP BY region_id",
            "SELECT region_id, AVG(order_amount) FROM fact_order GROUP BY region_id",
        )

        self.assertEqual("SEMANTIC_DRIFT", removed_filter["code"])
        self.assertEqual("SEMANTIC_DRIFT", changed_metric["code"])
        self.assertIn("聚合函数或DISTINCT口径发生变化", changed_metric["violations"])

    def test_filter_operator_and_join_type_changes_are_semantic_drift(self):
        changed_operator = self.evaluate(
            "SELECT order_id FROM fact_order WHERE order_amount >= 100",
            "SELECT order_id FROM fact_order WHERE order_amount <= 100",
        )
        changed_join = self.evaluate(
            "SELECT o.order_id FROM fact_order o LEFT JOIN dim_region r "
            "ON o.region_id = r.region_id",
            "SELECT o.order_id FROM fact_order o INNER JOIN dim_region r "
            "ON o.region_id = r.region_id",
        )

        self.assertEqual("SEMANTIC_DRIFT", changed_operator["code"])
        self.assertIn("过滤比较方式发生变化", changed_operator["violations"])
        self.assertEqual("SEMANTIC_DRIFT", changed_join["code"])
        self.assertIn("JOIN类型发生变化", changed_join["violations"])

    def test_unparseable_candidate_is_deferred_to_ast_audit(self):
        result = self.evaluate(
            "SELECT order_id FROM fact_order",
            "SELECT FROM",
        )

        self.assertTrue(result["passed"])
        self.assertEqual("DEFERRED_TO_AUDIT", result["code"])

    def test_guard_node_records_history_and_stops_graph(self):
        runtime = FakeRuntime()
        state = {
            "sql": "SELECT AVG(order_amount) FROM fact_order",
            "original_sql": "SELECT SUM(order_amount) FROM fact_order",
            "db_info": {"dialect": "mysql"},
            "retry_count": 1,
            "repair_history": [{
                "attempt": 1,
                "input_sql": "SELECT SUM(order_amount) FROM fact_order",
                "candidate_sql": "SELECT AVG(order_amount) FROM fact_order",
                "error": "unknown column",
                "input_fingerprint": "",
                "candidate_fingerprint": "",
                "guard_code": "PENDING",
                "guard_message": "等待修复保护检查",
            }],
        }

        update = repair_guard(state, runtime)

        self.assertEqual("SEMANTIC_DRIFT", update["repair_stop_reason"])
        self.assertEqual("SEMANTIC_DRIFT", update["repair_history"][-1]["guard_code"])
        self.assertEqual("end", route_after_repair_guard(update))
        self.assertEqual("SEMANTIC_DRIFT", json.loads(update["error"])["code"])
        self.assertTrue(any(event["type"] == "sql_repair_guard" for event in runtime.events))


class SemanticContextRegressionTests(unittest.TestCase):
    def test_time_context_always_contains_fact_date_join_column(self):
        table_infos = []

        _ensure_time_tables(table_infos)

        fact_order = next(table for table in table_infos if table["name"] == "fact_order")
        dim_date = next(table for table in table_infos if table["name"] == "dim_date")
        self.assertIn("date_id", {column["name"] for column in fact_order["columns"]})
        self.assertIn("date_id", {column["name"] for column in dim_date["columns"]})


if __name__ == "__main__":
    unittest.main()
