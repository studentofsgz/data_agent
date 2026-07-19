import json
import unittest
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from app.agent.confidence import evaluate_confidence
from app.agent.context import DataAgentContext
from app.agent.graph import (
    route_after_confidence_confirmation,
    route_after_confidence_guard,
)
from app.agent.nodes.confidence_guard import confidence_guard
from app.agent.nodes.confirm_confidence import confirm_confidence
from app.agent.query_intent import analyze_query_intent, extract_query_intent
from app.agent.state import DataAgentState
from app.services.query_service import QueryService


TABLE_INFOS = [
    {
        "name": "fact_order",
        "columns": [
            {"name": "order_amount"},
            {"name": "order_id"},
        ],
    },
    {
        "name": "dim_region",
        "columns": [{"name": "region_name"}],
    },
]


def confidence_inputs(
    question="统计各地区的GMV",
    *,
    degraded=False,
    exact=True,
    scores=True,
    table_infos=None,
):
    intent = extract_query_intent(question)
    semantic_metrics = [
        {"name": name}
        for name in intent["metrics"]
        if name in {"GMV", "AOV", "ORDER_COUNT", "SALES_QUANTITY"}
    ]
    return {
        "query_intent": intent,
        "table_infos": TABLE_INFOS if table_infos is None else table_infos,
        "metric_infos": [{"name": "GMV"}],
        "metric_semantics": {"metrics": semantic_metrics},
        "column_recall_sources": (
            {"fact_order.order_amount": ["exact_alias"]} if exact else {}
        ),
        "metric_recall_sources": {"GMV": ["exact_alias"]} if exact else {},
        "column_candidate_scores": (
            {"fact_order.order_amount": 0.8} if scores else {}
        ),
        "metric_candidate_scores": {"GMV": 0.8} if scores else {},
        "schema_linking_degraded": degraded,
    }


def medium_state():
    return confidence_inputs(degraded=True, exact=False, scores=False)


class FakeRuntime:
    def __init__(self):
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


def complete_for_test(state, runtime):
    runtime.stream_writer({"type": "test_completed"})
    return {}


def build_confidence_graph():
    builder = StateGraph(
        state_schema=DataAgentState,
        context_schema=DataAgentContext,
    )
    builder.add_node("confidence_guard", confidence_guard)
    builder.add_node("confirm_confidence", confirm_confidence)
    builder.add_node("complete", complete_for_test)
    builder.add_edge(START, "confidence_guard")
    builder.add_conditional_edges(
        "confidence_guard",
        route_after_confidence_guard,
        {
            "generate_sql": "complete",
            "confirm_confidence": "confirm_confidence",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "confirm_confidence",
        route_after_confidence_confirmation,
        {"generate_sql": "complete", "end": END},
    )
    builder.add_edge("complete", END)
    return builder.compile(checkpointer=InMemorySaver())


async def collect_events(graph, graph_input, config):
    return [
        event
        async for event in graph.astream(
            graph_input,
            config=config,
            context={},
            stream_mode="custom",
        )
    ]


async def collect_sse(generator):
    return [
        json.loads(item.removeprefix("data: ").strip())
        async for item in generator
    ]


class ConfidencePolicyTests(unittest.TestCase):
    def test_high_confidence_semantic_metric_proceeds(self):
        result = evaluate_confidence(**confidence_inputs())

        self.assertEqual("high", result["level"])
        self.assertEqual("proceed", result["action"])
        self.assertEqual("CONFIDENCE_HIGH", result["code"])
        self.assertGreaterEqual(result["score"], 0.75)

    def test_explicit_unknown_metric_reaches_guard_and_is_rejected(self):
        intent, ambiguity = analyze_query_intent("统计各地区的活跃用户增长率")
        result = evaluate_confidence(**confidence_inputs(
            "统计各地区的活跃用户增长率"
        ))

        self.assertFalse(ambiguity["needs_clarification"])
        self.assertEqual(["活跃用户", "增长率"], intent["unresolved_metric_mentions"])
        self.assertEqual("low", result["level"])
        self.assertEqual("reject", result["action"])
        self.assertEqual("UNKNOWN_METRIC", result["code"])
        self.assertLess(result["score"], 0.45)

    def test_missing_required_dimension_is_rejected(self):
        result = evaluate_confidence(**confidence_inputs(
            "统计各城市的GMV"
        ))

        self.assertEqual("MISSING_REQUIRED_DIMENSION", result["code"])
        self.assertIn("dim_region.city", result["evidence"]["missing_dimensions"])

    def test_degraded_linking_without_scores_requires_confirmation(self):
        result = evaluate_confidence(**medium_state())

        self.assertEqual("medium", result["level"])
        self.assertEqual("confirm", result["action"])
        self.assertEqual("CONFIRM_INTERPRETATION", result["code"])
        self.assertIn("是否按这个理解继续", result["question"])

    def test_no_schema_context_is_rejected(self):
        result = evaluate_confidence(**confidence_inputs(table_infos=[]))

        self.assertEqual("NO_SCHEMA_CONTEXT", result["code"])


class ConfidenceNodeTests(unittest.TestCase):
    def test_low_confidence_emits_structured_rejection(self):
        runtime = FakeRuntime()
        update = confidence_guard(
            confidence_inputs("统计各地区的活跃用户增长率"),
            runtime,
        )

        self.assertEqual("end", route_after_confidence_guard(update))
        rejection = next(
            event for event in runtime.events
            if event["type"] == "confidence_rejected"
        )
        self.assertEqual("UNKNOWN_METRIC", rejection["code"])


class ConfidenceHumanLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_medium_confidence_pauses_and_confirmation_continues(self):
        graph = build_confidence_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}

        first = await collect_events(graph, medium_state(), config)
        paused = await graph.aget_state(config)

        self.assertIn(
            "confidence_confirmation_required",
            [event["type"] for event in first],
        )
        self.assertEqual(("confirm_confidence",), paused.next)
        self.assertTrue(paused.tasks[0].interrupts)

        resumed = await collect_events(graph, Command(resume="确认"), config)
        completed = await graph.aget_state(config)

        self.assertIn("test_completed", [event["type"] for event in resumed])
        self.assertTrue(completed.values["confidence_confirmed"])
        self.assertEqual(
            "CONFIDENCE_CONFIRMED",
            completed.values["confidence_result"]["code"],
        )

    async def test_user_rejection_stops_before_generation(self):
        graph = build_confidence_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}
        await collect_events(graph, medium_state(), config)

        resumed = await collect_events(graph, Command(resume="不对"), config)
        completed = await graph.aget_state(config)

        self.assertNotIn("test_completed", [event["type"] for event in resumed])
        self.assertFalse(completed.values["confidence_confirmed"])
        self.assertEqual(
            "CONFIDENCE_REJECTED_BY_USER",
            completed.values["confidence_result"]["code"],
        )

    async def test_invalid_answers_stop_after_configured_attempts(self):
        graph = build_confidence_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}
        await collect_events(graph, medium_state(), config)

        await collect_events(graph, Command(resume="我不确定"), config)
        second_pause = await graph.aget_state(config)
        self.assertTrue(second_pause.tasks[0].interrupts)

        events = await collect_events(graph, Command(resume="再想想"), config)
        completed = await graph.aget_state(config)

        self.assertIn(
            "confidence_confirmation_invalid",
            [event["type"] for event in events],
        )
        self.assertFalse(completed.next)


class ConfidenceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_confidence_finishes_as_rejected_not_completed(self):
        graph = build_confidence_graph()
        service = QueryService(None, None, None, None, None, None, graph)

        events = await collect_sse(service.query(query="统计活跃用户增长率"))

        self.assertEqual("workflow_rejected", events[-1]["type"])
        self.assertEqual("NO_SCHEMA_CONTEXT", events[-1]["code"])
        self.assertNotIn("workflow_completed", [event["type"] for event in events])


if __name__ == "__main__":
    unittest.main()
