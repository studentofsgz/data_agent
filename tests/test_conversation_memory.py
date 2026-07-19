import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from app.agent.context import DataAgentContext
from app.agent.conversation_memory import (
    build_turn_input,
    resolve_structured_followup,
)
from app.agent.graph import route_after_ambiguity_guard
from app.agent.graph_runtime import GraphRuntime
from app.agent.nodes.ambiguity_guard import ambiguity_guard
from app.agent.nodes.context_manager import context_manager
from app.agent.nodes.remember_turn import remember_turn
from app.agent.query_intent import extract_query_intent
from app.agent.state import DataAgentState
from app.services.query_service import QueryService
from tests.test_clarification_resume import collect_events, collect_sse


class FakeRuntime:
    def __init__(self):
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


def fake_execute(state, runtime):
    runtime.stream_writer({"type": "test_completed", "query": state["query"]})
    return {
        "sql": "SELECT 1",
        "result_summary": {
            "row_count": 1,
            "columns": ["gmv"],
            "preview": [{"gmv": 100}],
            "truncated": False,
        },
    }


def build_conversation_graph(checkpointer=None):
    builder = StateGraph(
        state_schema=DataAgentState,
        context_schema=DataAgentContext,
    )
    builder.add_node("context_manager", context_manager)
    builder.add_node("ambiguity_guard", ambiguity_guard)
    builder.add_node("fake_execute", fake_execute)
    builder.add_node("remember_turn", remember_turn)
    builder.add_edge(START, "context_manager")
    builder.add_edge("context_manager", "ambiguity_guard")
    builder.add_conditional_edges(
        "ambiguity_guard",
        route_after_ambiguity_guard,
        {
            "clarify_intent": END,
            "extract_keywords": "fake_execute",
            "end": END,
        },
    )
    builder.add_edge("fake_execute", "remember_turn")
    builder.add_edge("remember_turn", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class StructuredConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.previous = extract_query_intent("统计2025年各地区的GMV")

    def test_region_follow_up_inherits_metric_time_and_dimension(self):
        result = resolve_structured_followup("那华东呢？", self.previous)

        self.assertTrue(result["applied"])
        self.assertEqual("structured_memory", result["strategy"])
        self.assertIn("2025年", result["query_after"])
        self.assertIn("华东地区", result["query_after"])
        self.assertIn("统计销售额", result["query_after"])
        self.assertNotIn("按年展示", result["query_after"])
        self.assertIn("metrics", result["inherited_slots"])
        self.assertIn("filter.region", result["overridden_slots"])

    def test_metric_follow_up_overrides_metric_and_keeps_other_slots(self):
        result = resolve_structured_followup("改成销量", self.previous)

        self.assertTrue(result["applied"])
        self.assertIn("统计销量", result["query_after"])
        self.assertNotIn("统计销售额", result["query_after"])
        self.assertIn("metrics", result["overridden_slots"])
        self.assertIn("dimensions", result["inherited_slots"])

    def test_independent_query_does_not_inherit_old_context(self):
        result = resolve_structured_followup(
            "统计今年各城市的GMV",
            self.previous,
        )

        self.assertFalse(result["applied"])
        self.assertEqual("none", result["strategy"])

    def test_unknown_follow_up_is_delegated_to_llm_fallback(self):
        result = resolve_structured_followup("那利润呢？", self.previous)

        self.assertFalse(result["applied"])
        self.assertEqual("llm_fallback_required", result["strategy"])

    def test_explicit_daily_grain_is_inherited(self):
        previous = extract_query_intent("统计2025年1月每天的销售额")
        result = resolve_structured_followup("那华东呢？", previous)

        self.assertIn("按天展示", result["query_after"])
        self.assertIn("time.grain", result["inherited_slots"])

    def test_conversation_history_is_bounded(self):
        old_history = [
            {
                "turn": turn,
                "raw_query": f"q{turn}",
                "resolved_query": f"q{turn}",
                "intent": {},
                "sql": "SELECT 1",
                "result_summary": {},
                "completed_at": "2026-01-01T00:00:00+00:00",
            }
            for turn in range(1, 7)
        ]

        update = remember_turn(
            {
                "query": "q7",
                "raw_query": "q7",
                "query_intent": {},
                "sql": "SELECT 1",
                "conversation_turn": 6,
                "conversation_history": old_history,
                "result_summary": {},
            },
            FakeRuntime(),
        )

        self.assertEqual(6, len(update["conversation_history"]))
        self.assertEqual(2, update["conversation_history"][0]["turn"])
        self.assertEqual(7, update["conversation_history"][-1]["turn"])


class ConversationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_request_reuses_structured_state_in_same_thread(self):
        graph = build_conversation_graph()
        service = QueryService(None, None, None, None, None, None, graph)

        first = await collect_sse(service.query(query="统计2025年各地区的GMV"))
        thread_id = first[0]["thread_id"]
        self.assertEqual("new", first[0]["mode"])
        self.assertEqual(1, first[-1]["conversation_turn"])

        second = await collect_sse(
            service.query(query="那华东呢？", thread_id=thread_id)
        )
        resolution = next(e for e in second if e["type"] == "context_resolution")
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})

        self.assertEqual("follow_up", second[0]["mode"])
        self.assertEqual("structured_memory", resolution["strategy"])
        self.assertIn("2025年", state.values["query"])
        self.assertIn("华东地区", state.values["query"])
        self.assertEqual(2, state.values["conversation_turn"])
        self.assertEqual(2, len(state.values["conversation_history"]))
        self.assertEqual(1, state.values["last_result_summary"]["row_count"])

    async def test_new_query_cannot_overwrite_paused_clarification(self):
        from tests.test_clarification_resume import build_clarification_graph

        graph = build_clarification_graph()
        service = QueryService(None, None, None, None, None, None, graph)
        first = await collect_sse(service.query(query="1月份每天的销售额"))
        thread_id = first[0]["thread_id"]

        second = await collect_sse(
            service.query(query="统计今年的GMV", thread_id=thread_id)
        )

        self.assertEqual(1, len(second))
        self.assertEqual("WORKFLOW_PAUSED", second[0]["code"])

    async def test_invalid_thread_id_is_rejected(self):
        graph = build_conversation_graph()
        service = QueryService(None, None, None, None, None, None, graph)

        events = await collect_sse(
            service.query(query="统计GMV", thread_id="bad/thread")
        )

        self.assertEqual("INVALID_THREAD_ID", events[0]["code"])

    async def test_resume_requires_original_thread_id(self):
        graph = build_conversation_graph()
        service = QueryService(None, None, None, None, None, None, graph)

        events = await collect_sse(service.query(resume="2025年"))

        self.assertEqual("THREAD_ID_REQUIRED_FOR_RESUME", events[0]["code"])

    def test_session_expiration_uses_most_recent_activity(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=2)).isoformat()
        recent = (now - timedelta(minutes=1)).isoformat()

        self.assertTrue(QueryService._session_expired({"last_completed_at": old}))
        self.assertFalse(QueryService._session_expired({
            "last_completed_at": old,
            "turn_started_at": recent,
        }))


class DurableCheckpointerTests(unittest.IsolatedAsyncioTestCase):
    async def test_paused_workflow_survives_runtime_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.sqlite"
            config = {"configurable": {"thread_id": f"restart-{uuid4().hex}"}}

            first_runtime = GraphRuntime(path, persistent=True)
            await first_runtime.start()
            first_graph = first_runtime.get_graph()
            await collect_events(
                first_graph,
                build_turn_input("1月份每天的销售额"),
                config,
            )
            paused = await first_graph.aget_state(config)
            self.assertTrue(paused.tasks[0].interrupts)
            await first_runtime.close()

            second_runtime = GraphRuntime(path, persistent=True)
            await second_runtime.start()
            second_graph = second_runtime.get_graph()
            restored = await second_graph.aget_state(config)
            self.assertTrue(restored.tasks[0].interrupts)

            events = await collect_events(
                second_graph,
                Command(resume="取消"),
                config,
            )
            completed = await second_graph.aget_state(config)
            self.assertIn("clarification_cancelled", [e["type"] for e in events])
            self.assertFalse(completed.next)
            await second_runtime.close()

    async def test_completed_memory_survives_restart_and_resolves_follow_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.sqlite"
            thread_id = f"memory-{uuid4().hex}"

            async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
                await saver.setup()
                first_graph = build_conversation_graph(saver)
                first_service = QueryService(
                    None, None, None, None, None, None, first_graph
                )
                first = await collect_sse(
                    first_service.query(
                        query="统计2025年各地区的GMV",
                        thread_id=thread_id,
                    )
                )
                self.assertEqual(1, first[-1]["conversation_turn"])

            async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
                await saver.setup()
                second_graph = build_conversation_graph(saver)
                second_service = QueryService(
                    None, None, None, None, None, None, second_graph
                )
                second = await collect_sse(
                    second_service.query(
                        query="那华东呢？",
                        thread_id=thread_id,
                    )
                )
                resolution = next(
                    event for event in second
                    if event["type"] == "context_resolution"
                )
                state = await second_graph.aget_state({
                    "configurable": {"thread_id": thread_id}
                })

                self.assertEqual("structured_memory", resolution["strategy"])
                self.assertIn("2025年", state.values["query"])
                self.assertIn("华东地区", state.values["query"])
                self.assertEqual(2, state.values["conversation_turn"])


if __name__ == "__main__":
    unittest.main()
