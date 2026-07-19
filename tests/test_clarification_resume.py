import json
import unittest
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from app.agent.context import DataAgentContext
from app.agent.graph import (
    route_after_ambiguity_guard,
    route_after_clarification,
)
from app.agent.nodes.ambiguity_guard import ambiguity_guard
from app.agent.nodes.clarify_intent import clarify_intent
from app.agent.state import DataAgentState
from app.services.query_service import QueryService


def complete_for_test(state, runtime):
    runtime.stream_writer({"type": "test_completed", "query": state["query"]})
    return {}


def build_clarification_graph():
    builder = StateGraph(
        state_schema=DataAgentState,
        context_schema=DataAgentContext,
    )
    builder.add_node("ambiguity_guard", ambiguity_guard)
    builder.add_node("clarify_intent", clarify_intent)
    builder.add_node("complete", complete_for_test)
    builder.add_edge(START, "ambiguity_guard")
    builder.add_conditional_edges(
        "ambiguity_guard",
        route_after_ambiguity_guard,
        {
            "clarify_intent": "clarify_intent",
            "extract_keywords": "complete",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "clarify_intent",
        route_after_clarification,
        {"ambiguity_guard": "ambiguity_guard", "end": END},
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


class ClarificationResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_answer_resumes_same_workflow(self):
        graph = build_clarification_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}

        first_events = await collect_events(
            graph,
            {"query": "1月份每天的销售额", "messages": []},
            config,
        )
        paused = await graph.aget_state(config)

        self.assertIn("clarification_required", [e["type"] for e in first_events])
        self.assertEqual(("clarify_intent",), paused.next)
        self.assertEqual(1, len(paused.tasks[0].interrupts))

        resumed_events = await collect_events(
            graph,
            Command(resume="2025年"),
            config,
        )
        completed = await graph.aget_state(config)

        self.assertIn("clarification_resumed", [e["type"] for e in resumed_events])
        self.assertIn("test_completed", [e["type"] for e in resumed_events])
        self.assertEqual(2025, completed.values["query_intent"]["time"]["year"])
        self.assertEqual(1, completed.values["clarification_rounds"])
        self.assertEqual(1, len(completed.values["clarification_history"]))
        self.assertFalse(completed.next)

    async def test_multiple_missing_slots_are_clarified_one_at_a_time(self):
        graph = build_clarification_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}

        await collect_events(
            graph,
            {"query": "最近销售情况怎么样", "messages": []},
            config,
        )
        first_pause = await graph.aget_state(config)
        first_interrupt = first_pause.tasks[0].interrupts[0].value
        self.assertEqual("metric", first_interrupt["asked_slot"])

        await collect_events(graph, Command(resume="GMV"), config)
        second_pause = await graph.aget_state(config)
        second_interrupt = second_pause.tasks[0].interrupts[0].value
        self.assertEqual("time.range", second_interrupt["asked_slot"])

        final_events = await collect_events(
            graph,
            Command(resume="最近30天"),
            config,
        )
        completed = await graph.aget_state(config)

        self.assertIn("test_completed", [e["type"] for e in final_events])
        self.assertEqual(2, completed.values["clarification_rounds"])
        self.assertEqual(2, len(completed.values["clarification_history"]))
        self.assertIn("GMV", completed.values["query"])
        self.assertIn("最近30天", completed.values["query"])
        self.assertFalse(completed.values["ambiguity_result"]["needs_clarification"])

    async def test_thread_ids_keep_paused_state_isolated(self):
        graph = build_clarification_graph()
        config_a = {"configurable": {"thread_id": f"a-{uuid4().hex}"}}
        config_b = {"configurable": {"thread_id": f"b-{uuid4().hex}"}}

        await collect_events(
            graph,
            {"query": "1月份每天的销售额", "messages": []},
            config_a,
        )
        await collect_events(
            graph,
            {"query": "2月份每天的销售额", "messages": []},
            config_b,
        )
        await collect_events(graph, Command(resume="2024年"), config_a)
        await collect_events(graph, Command(resume="2025年"), config_b)

        state_a = await graph.aget_state(config_a)
        state_b = await graph.aget_state(config_b)

        self.assertEqual(2024, state_a.values["query_intent"]["time"]["year"])
        self.assertEqual(1, state_a.values["query_intent"]["time"]["month"])
        self.assertEqual(2025, state_b.values["query_intent"]["time"]["year"])
        self.assertEqual(2, state_b.values["query_intent"]["time"]["month"])

    async def test_user_can_cancel_a_paused_workflow(self):
        graph = build_clarification_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}

        await collect_events(
            graph,
            {"query": "1月份每天的销售额", "messages": []},
            config,
        )
        events = await collect_events(graph, Command(resume="取消"), config)
        completed = await graph.aget_state(config)

        self.assertIn("clarification_cancelled", [e["type"] for e in events])
        self.assertTrue(completed.values["clarification_cancelled"])
        self.assertFalse(completed.next)

    async def test_unresolved_answer_stops_at_round_limit(self):
        graph = build_clarification_graph()
        config = {"configurable": {"thread_id": uuid4().hex}}

        await collect_events(
            graph,
            {"query": "1月份每天的销售额", "messages": []},
            config,
        )
        await collect_events(graph, Command(resume="不知道"), config)
        await collect_events(graph, Command(resume="还是不知道"), config)
        events = await collect_events(graph, Command(resume="无法确定"), config)
        completed = await graph.aget_state(config)

        self.assertIn("clarification_limit_reached", [e["type"] for e in events])
        self.assertEqual(3, completed.values["clarification_rounds"])
        self.assertEqual(
            "CLARIFICATION_LIMIT_REACHED",
            completed.values["ambiguity_result"]["code"],
        )
        self.assertFalse(completed.next)


class QueryServiceResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = QueryService(None, None, None, None, None, None)

    async def test_service_returns_thread_id_and_resumes_cancel(self):
        first_events = await collect_sse(
            self.service.query(query="1月份每天的销售额")
        )
        started = next(e for e in first_events if e["type"] == "workflow_started")
        workflow_id = started["thread_id"]

        self.assertIn("clarification_required", [e["type"] for e in first_events])
        self.assertIn("workflow_paused", [e["type"] for e in first_events])
        self.assertTrue(
            all(e.get("thread_id") == workflow_id for e in first_events)
        )

        resumed_events = await collect_sse(
            self.service.query(thread_id=workflow_id, resume="取消")
        )

        self.assertEqual("workflow_resuming", resumed_events[0]["type"])
        self.assertIn("clarification_cancelled", [e["type"] for e in resumed_events])
        self.assertEqual("workflow_completed", resumed_events[-1]["type"])

    async def test_service_rejects_unknown_resume_thread(self):
        events = await collect_sse(
            self.service.query(
                thread_id=f"missing-{uuid4().hex}",
                resume="2025年",
            )
        )

        self.assertEqual(1, len(events))
        self.assertEqual("RESUME_NOT_AVAILABLE", events[0]["code"])

    async def test_service_requires_query_for_new_workflow(self):
        events = await collect_sse(self.service.query())

        self.assertEqual(1, len(events))
        self.assertEqual("QUERY_REQUIRED", events[0]["code"])


if __name__ == "__main__":
    unittest.main()
