import asyncio
import json
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from app.agent.observability import current_node_name, instrument_node
from eval.llm_tracking import LLMCallTracker
from eval.metrics import aggregate_results, compare_result_rows, evaluate_case
from eval.runner import load_cases, load_goldens, merge_goldens


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResultComparisonTests(unittest.TestCase):
    def test_numeric_tolerance_and_metric_aliases(self):
        matched, diff = compare_result_rows(
            [{"region_name": "华东", "销售总额": Decimal("100.005")}],
            [{"region_name": "华东", "total_amount": 100.0}],
            abs_tolerance=0.01,
        )

        self.assertTrue(matched)
        self.assertIsNone(diff)

    def test_unordered_results_ignore_row_order(self):
        matched, _ = compare_result_rows(
            [{"province": "浙江省", "销量": 10}, {"province": "广东省", "销量": 20}],
            [
                {"province": "广东省", "total_quantity": 20},
                {"province": "浙江省", "total_quantity": 10},
            ],
        )

        self.assertTrue(matched)

    def test_ordered_results_report_value_difference(self):
        matched, diff = compare_result_rows(
            [{"province": "浙江省", "销量": 10}, {"province": "广东省", "销量": 20}],
            [
                {"province": "广东省", "total_quantity": 20},
                {"province": "浙江省", "total_quantity": 10},
            ],
            ordered=True,
        )

        self.assertFalse(matched)
        self.assertEqual(diff["reason"], "value_mismatch")

    def test_empty_aggregate_row_counts_as_no_data(self):
        result = evaluate_case(
            case={
                "id": "empty",
                "question": "没有数据的聚合",
                "expect_not_empty": False,
                "expected_result": [{"total_amount": None}],
            },
            sql="SELECT SUM(order_amount) AS total_amount FROM fact_order",
            rows=[{"total_amount": None}],
            error="",
            elapsed_seconds=0.01,
            correction_attempts=0,
            event_count=1,
            result_received=True,
        )

        self.assertTrue(result["not_empty_ok"])
        self.assertTrue(result["expected_result_ok"])

    def test_unconfigured_expected_result_is_na(self):
        result = evaluate_case(
            case={"id": "plain", "question": "普通问题"},
            sql="SELECT 1",
            rows=[{"1": 1}],
            error="",
            elapsed_seconds=0.01,
            correction_attempts=0,
            event_count=1,
            result_received=True,
        )
        summary = aggregate_results([result])["summary"]

        self.assertEqual(summary["expected_result_ok"]["total"], 0)
        self.assertIsNone(summary["expected_result_ok"]["rate"])


class ObservabilityTests(unittest.TestCase):
    def test_node_wrapper_emits_start_and_completion(self):
        events = []
        runtime = SimpleNamespace(stream_writer=events.append)

        async def sample_node(state, runtime):
            del runtime
            self.assertEqual(current_node_name.get(), "sample_node")
            return {"value": state["value"] + 1}

        result = asyncio.run(
            instrument_node("sample_node", sample_node)({"value": 1}, runtime)
        )

        self.assertEqual(result, {"value": 2})
        self.assertEqual([event["status"] for event in events], ["running", "success"])
        self.assertEqual(events[0]["invocation_id"], events[1]["invocation_id"])
        self.assertGreaterEqual(events[1]["elapsed_seconds"], 0)
        self.assertIsNone(current_node_name.get())

    def test_llm_tracker_records_node_latency_and_tokens(self):
        tracker = LLMCallTracker()
        run_id = uuid4()
        tracker.on_chat_model_start(
            {"name": "test-model", "kwargs": {}},
            [[SimpleNamespace(content="问题")]],
            run_id=run_id,
            metadata={"langgraph_node": "generate_sql"},
        )
        message = SimpleNamespace(
            content="SELECT 1",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
            },
            response_metadata={"model_name": "test-model"},
        )
        response = SimpleNamespace(
            generations=[[SimpleNamespace(message=message)]],
            llm_output={},
        )
        tracker.on_llm_end(response, run_id=run_id)

        calls = tracker.snapshot()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["node"], "generate_sql")
        self.assertEqual(calls[0]["total_tokens"], 13)
        self.assertTrue(calls[0]["usage_reported"])

    def test_langgraph_propagates_node_name_to_llm_tracker(self):
        class MiniState(TypedDict):
            query: str
            answer: str

        model = FakeListChatModel(responses=["ok"])

        async def model_node(state: MiniState, runtime: Runtime):
            response = await model.ainvoke(state["query"])
            runtime.stream_writer({"type": "answer", "value": response.content})
            return {"answer": response.content}

        builder = StateGraph(MiniState)
        builder.add_node(
            "model_node",
            instrument_node("model_node", model_node),
        )
        builder.add_edge(START, "model_node")
        builder.add_edge("model_node", END)
        graph = builder.compile()

        async def run_graph():
            tracker = LLMCallTracker()
            events = []
            async for event in graph.astream(
                {"query": "hi", "answer": ""},
                config={"callbacks": [tracker]},
                stream_mode="custom",
            ):
                events.append(event)
            return tracker.snapshot(), events

        calls, events = asyncio.run(run_graph())
        timing_statuses = [
            event["status"]
            for event in events
            if event.get("type") == "node_timing"
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["node"], "model_node")
        self.assertEqual(timing_statuses, ["running", "success"])

    def test_observability_is_aggregated(self):
        result = evaluate_case(
            case={"id": "observed", "question": "测试"},
            sql="SELECT 1",
            rows=[{"1": 1}],
            error="",
            elapsed_seconds=2.0,
            correction_attempts=0,
            event_count=4,
            result_received=True,
            node_timings=[
                {
                    "node": "generate_sql",
                    "status": "success",
                    "elapsed_seconds": 1.5,
                }
            ],
            llm_calls=[
                {
                    "node": "generate_sql",
                    "status": "success",
                    "elapsed_seconds": 1.2,
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "usage_reported": True,
                }
            ],
            sql_cache_hit=False,
        )
        observability = aggregate_results([result])["summary"]["observability"]

        self.assertEqual(result["slowest_node"]["node"], "generate_sql")
        self.assertEqual(observability["node_timings"]["generate_sql"]["count"], 1)
        self.assertEqual(observability["llm"]["count"], 1)
        self.assertEqual(observability["llm"]["total_tokens"], 13)
        self.assertEqual(observability["sql_cache"]["misses"], 1)


class GoldenResultTests(unittest.TestCase):
    def test_twenty_goldens_are_valid_and_mergeable(self):
        cases = load_cases(PROJECT_ROOT / "eval" / "data" / "questions.json")
        goldens = load_goldens(
            PROJECT_ROOT / "eval" / "data" / "golden_results.json"
        )
        merged = merge_goldens(cases, goldens)
        golden_cases = [
            case for case in merged if case.get("expected_result") is not None
        ]

        self.assertEqual(len(goldens), 20)
        self.assertEqual(len(golden_cases), 20)
        self.assertEqual(len({case["id"] for case in golden_cases}), 20)
        for case in golden_cases:
            self.assertTrue(case.get("gold_sql", "").upper().startswith("SELECT"))
            self.assertIsInstance(case["expected_result"], list)

    def test_golden_file_is_json_serializable(self):
        with (PROJECT_ROOT / "eval" / "data" / "golden_results.json").open(
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
