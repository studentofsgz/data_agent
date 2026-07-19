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
from eval.schema_linking import derive_gold_schema


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

    def test_repair_guard_stop_reason_is_aggregated(self):
        result = evaluate_case(
            case={"id": "repair-loop", "question": "测试修复循环"},
            sql="SELECT order_id FROM fact_order",
            rows=[],
            error="repair stopped",
            elapsed_seconds=0.01,
            correction_attempts=2,
            event_count=2,
            result_received=False,
            repair_guard_events=[{
                "status": "stopped",
                "code": "REPAIR_CYCLE",
                "message": "loop",
                "violations": [],
                "attempt": 2,
            }],
        )

        summary = aggregate_results([result])["summary"]
        self.assertEqual("REPAIR_CYCLE", result["repair_stop_reason"])
        self.assertEqual(1, summary["repair_guard_stopped_cases"])
        self.assertEqual({"REPAIR_CYCLE": 1}, summary["repair_stop_reasons"])

    def test_query_cost_and_sandbox_metrics_are_aggregated(self):
        result = evaluate_case(
            case={"id": "cost", "question": "成本治理测试"},
            sql="SELECT order_id FROM fact_order LIMIT 10",
            rows=[{"order_id": "o1"}],
            error="",
            elapsed_seconds=0.02,
            correction_attempts=0,
            event_count=2,
            result_received=True,
            query_plan_events=[{
                "status": "passed",
                "code": "PLAN_OK",
                "estimated_rows": 100,
                "warnings": ["USING_FILESORT"],
            }],
            sql_sandbox_events=[{
                "status": "success",
                "returned_rows": 1,
                "truncated": False,
            }],
        )

        summary = aggregate_results([result])["summary"]["query_governance"]
        self.assertEqual(1, summary["plan_passed"])
        self.assertEqual(100.0, summary["avg_estimated_rows"])
        self.assertEqual({"USING_FILESORT": 1}, summary["warning_codes"])
        self.assertEqual(1, summary["sandbox_executions"])

    def test_clarification_precision_recall_and_code_are_aggregated(self):
        cases = [
            evaluate_case(
                case={
                    "id": "ambiguous",
                    "question": "1月份销售额",
                    "expect_clarification": True,
                    "expect_clarification_code": "MISSING_YEAR_FOR_MONTH",
                },
                sql="",
                rows=[],
                error="",
                elapsed_seconds=0.01,
                correction_attempts=0,
                event_count=2,
                result_received=False,
                clarification_event={
                    "code": "MISSING_YEAR_FOR_MONTH",
                    "question": "哪一年？",
                },
            ),
            evaluate_case(
                case={
                    "id": "clear",
                    "question": "2025年1月销售额",
                    "expect_clarification": False,
                },
                sql="SELECT 1",
                rows=[{"1": 1}],
                error="",
                elapsed_seconds=0.01,
                correction_attempts=0,
                event_count=1,
                result_received=True,
            ),
        ]

        summary = aggregate_results(cases)["summary"]["clarification"]
        self.assertEqual(100.0, summary["accuracy"]["rate"])
        self.assertEqual(100.0, summary["precision"])
        self.assertEqual(100.0, summary["recall"])
        self.assertEqual(1, summary["true_positive"])
        self.assertEqual(1, summary["true_negative"])

    def test_confidence_action_and_rejection_code_are_aggregated(self):
        result = evaluate_case(
            case={
                "id": "confidence-low",
                "question": "统计各地区的活跃用户增长率",
                "expect_confidence_action": "reject",
                "expect_confidence_code": "UNKNOWN_METRIC",
            },
            sql="",
            rows=[],
            error="",
            elapsed_seconds=0.01,
            correction_attempts=0,
            event_count=2,
            result_received=False,
            confidence_events=[{
                "type": "confidence_assessment",
                "level": "low",
                "action": "reject",
                "code": "UNKNOWN_METRIC",
                "score": 0.44,
            }],
        )

        summary = aggregate_results([result])["summary"]["confidence"]
        self.assertTrue(result["confidence_ok"])
        self.assertEqual(100.0, summary["accuracy"]["rate"])
        self.assertEqual({"low": 1}, summary["levels"])
        self.assertEqual({"reject": 1}, summary["actions"])
        self.assertEqual({"UNKNOWN_METRIC": 1}, summary["rejection_codes"])

    def test_grounded_answer_verification_and_fallback_are_aggregated(self):
        events = [
            {
                "type": "grounded_answer",
                "status": "generated",
                "fallback_reason": "",
                "verification": {
                    "passed": True,
                    "model_output_passed": True,
                    "invalid_numbers": [],
                },
            },
            {
                "type": "grounded_answer",
                "status": "fallback",
                "fallback_reason": "ungrounded_numeric_claim",
                "verification": {
                    "passed": True,
                    "model_output_passed": False,
                    "invalid_numbers": ["999999"],
                },
            },
        ]
        results = [
            evaluate_case(
                case={
                    "id": f"answer-{index}",
                    "question": "各地区GMV",
                    "expect_answer_status": event["status"],
                    "expect_answer_grounded": True,
                },
                sql="SELECT 1",
                rows=[{"gmv": 1}],
                error="",
                elapsed_seconds=0.01,
                correction_attempts=0,
                event_count=2,
                result_received=True,
                grounded_answer_event=event,
            )
            for index, event in enumerate(events)
        ]

        summary = aggregate_results(results)["summary"]["grounded_answer"]
        self.assertTrue(all(result["answer_ok"] for result in results))
        self.assertEqual(100.0, summary["accuracy"]["rate"])
        self.assertEqual(100.0, summary["final_grounded"]["rate"])
        self.assertEqual(50.0, summary["model_output_grounded"]["rate"])
        self.assertEqual(1, summary["caught_ungrounded_cases"])
        self.assertEqual({"fallback": 1, "generated": 1}, summary["statuses"])

    def test_schema_linking_metrics_are_derived_from_golden_sql(self):
        gold_sql = (
            "SELECT r.region_name, SUM(f.order_amount) AS gmv "
            "FROM fact_order f JOIN dim_region r "
            "ON f.region_id = r.region_id GROUP BY r.region_name"
        )
        derived = derive_gold_schema(gold_sql)
        self.assertEqual(["dim_region", "fact_order"], derived["tables"])
        self.assertIn("fact_order.order_amount", derived["columns"])
        self.assertIn(
            "dim_region.region_id=fact_order.region_id",
            derived["join_keys"],
        )
        self.assertEqual(["gmv"], derived["metrics"])

        result = evaluate_case(
            case={
                "id": "schema",
                "question": "各地区GMV",
                "expect_tables": ["fact_order", "dim_region"],
                "gold_sql": gold_sql,
            },
            sql=gold_sql,
            rows=[],
            error="",
            elapsed_seconds=0.01,
            correction_attempts=0,
            event_count=5,
            result_received=True,
            schema_linking_events=[
                {
                    "stage": "column_recall",
                    "candidates": [
                        {"id": "fact_order.order_amount", "sources": ["vector"]},
                        {"id": "dim_region.region_name", "sources": ["exact_alias"]},
                    ],
                },
                {
                    "stage": "metric_recall",
                    "candidates": [{"id": "GMV", "sources": ["exact_alias"]}],
                },
                {
                    "stage": "rerank",
                    "columns": [
                        {"id": "fact_order.order_amount"},
                        {"id": "dim_region.region_name"},
                    ],
                    "metrics": [{"id": "GMV"}],
                },
                {
                    "stage": "table_filter",
                    "tables": ["fact_order", "dim_region"],
                    "columns": derived["columns"],
                },
                {"stage": "metric_filter", "metrics": ["GMV"]},
            ],
        )
        summary = aggregate_results([result])["summary"]["schema_linking"]

        self.assertEqual(100.0, result["schema_linking"]["table_recall"]["rate"])
        self.assertEqual(100.0, result["schema_linking"]["join_key_coverage"]["rate"])
        self.assertEqual(100.0, summary["final_column_recall"]["rate"])
        self.assertEqual(100.0, summary["final_metric_recall"]["rate"])
        self.assertEqual(
            100.0,
            summary["metric_recall_by_source"]["exact_alias"]["rate"],
        )


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
