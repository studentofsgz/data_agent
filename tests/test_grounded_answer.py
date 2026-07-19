import asyncio
import json
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.answer_grounding import (
    build_provenance,
    deterministic_answer,
    sanitize_answer_rows,
    validate_numeric_grounding,
)
from app.agent.nodes.generate_answer import generate_answer


class FakeRuntime:
    def __init__(self):
        self.context = {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


def answer_state(rows=None, *, row_count=None, truncated=False):
    rows = rows if rows is not None else [
        {"region_name": "华东", "gmv": 107373},
        {"region_name": "华南", "gmv": 70202},
    ]
    return {
        "query": "统计2025年各地区的GMV",
        "query_intent": {
            "metrics": ["GMV"],
            "dimensions": ["region"],
            "time": {"year": 2025},
            "filters": [],
        },
        "sql": "SELECT region_name, SUM(order_amount) AS gmv FROM fact_order",
        "audit_result": {
            "tables": ["fact_order", "dim_region"],
            "columns": ["dim_region.region_name", "fact_order.order_amount"],
        },
        "metric_semantics": {
            "metrics": [{
                "name": "GMV",
                "display_name": "销售额",
                "expression": "SUM(fact_order.order_amount)",
            }],
        },
        "metric_infos": [],
        "answer_rows": rows,
        "result_summary": {
            "row_count": len(rows) if row_count is None else row_count,
            "columns": list(rows[0]) if rows else [],
            "preview": rows[:3],
            "truncated": truncated,
        },
    }


class AnswerGroundingPolicyTests(unittest.TestCase):
    def test_rows_are_bounded_and_long_cells_are_truncated(self):
        rows = [
            {"name": "a" * 20, "value": index}
            for index in range(5)
        ]

        sanitized = sanitize_answer_rows(
            rows,
            max_rows=2,
            max_cell_chars=5,
        )

        self.assertEqual(2, len(sanitized))
        self.assertEqual("aaaaa…", sanitized[0]["name"])

    def test_numeric_validation_accepts_rows_and_query_numbers(self):
        result = validate_numeric_grounding(
            texts=["2025年华东销售额为107,373，华南为70202。"],
            rows=answer_state()["answer_rows"],
            query="统计2025年各地区的GMV",
            row_count=2,
            tolerance=0.01,
        )

        self.assertTrue(result["passed"])
        self.assertEqual([], result["invalid_numbers"])

    def test_numeric_validation_rejects_invented_number(self):
        result = validate_numeric_grounding(
            texts=["华东销售额为999999。"],
            rows=answer_state()["answer_rows"],
            query="统计各地区的GMV",
            row_count=2,
            tolerance=0.01,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(["999999"], result["invalid_numbers"])

    def test_fallback_only_copies_result_values(self):
        answer = deterministic_answer(
            answer_state()["answer_rows"],
            row_count=2,
            max_rows=5,
        )

        self.assertIn("107373", answer)
        self.assertIn("70202", answer)
        self.assertNotIn("增长", answer)

    def test_provenance_contains_sql_schema_metric_and_row_scope(self):
        provenance = build_provenance(answer_state(), rows_used=2)

        self.assertIn("fact_order", provenance["tables"])
        self.assertEqual("GMV", provenance["metrics"][0]["name"])
        self.assertEqual(2, provenance["rows_used"])
        self.assertIn("SELECT", provenance["sql"])


class GroundedAnswerNodeTests(unittest.TestCase):
    def test_valid_model_answer_is_emitted_with_provenance(self):
        model = FakeListChatModel(responses=[json.dumps({
            "answer": "2025年华东GMV为107,373，华南为70,202。",
            "highlights": ["华东为107373"],
            "caveats": [],
        }, ensure_ascii=False)])
        runtime = FakeRuntime()

        with patch("app.agent.nodes.generate_answer.llm", model):
            update = asyncio.run(generate_answer(answer_state(), runtime))

        self.assertEqual("generated", update["answer_result"]["status"])
        self.assertTrue(update["answer_result"]["verification"]["passed"])
        self.assertEqual(
            "grounded_answer",
            next(e for e in runtime.events if e["type"] == "grounded_answer")["type"],
        )

    def test_invented_model_number_uses_deterministic_fallback(self):
        model = FakeListChatModel(responses=[json.dumps({
            "answer": "华东GMV为999999。",
            "highlights": [],
            "caveats": [],
        }, ensure_ascii=False)])
        runtime = FakeRuntime()

        with patch("app.agent.nodes.generate_answer.llm", model):
            update = asyncio.run(generate_answer(answer_state(), runtime))

        result = update["answer_result"]
        self.assertEqual("fallback", result["status"])
        self.assertEqual("ungrounded_numeric_claim", result["fallback_reason"])
        self.assertEqual(["999999"], result["verification"]["invalid_numbers"])
        self.assertNotIn("999999", result["answer"])

    def test_empty_result_skips_model_and_returns_grounded_message(self):
        runtime = FakeRuntime()
        update = asyncio.run(generate_answer(
            answer_state([], row_count=0),
            runtime,
        ))

        result = update["answer_result"]
        self.assertEqual("empty", result["status"])
        self.assertEqual("没有查询到符合条件的数据。", result["answer"])
        self.assertTrue(result["verification"]["passed"])

    def test_limited_answer_rows_add_scope_caveat(self):
        model = FakeListChatModel(responses=[json.dumps({
            "answer": "华东GMV为107373，华南为70202。",
            "highlights": [],
            "caveats": [],
        }, ensure_ascii=False)])
        runtime = FakeRuntime()
        state = answer_state(row_count=30, truncated=True)

        with patch("app.agent.nodes.generate_answer.llm", model):
            update = asyncio.run(generate_answer(state, runtime))

        caveats = update["answer_result"]["caveats"]
        self.assertTrue(any("执行上限" in item for item in caveats))
        self.assertTrue(any("前2行" in item for item in caveats))


if __name__ == "__main__":
    unittest.main()
