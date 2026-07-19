import unittest

from app.agent.graph import route_after_ambiguity_guard
from app.agent.nodes.ambiguity_guard import ambiguity_guard
from app.agent.query_intent import analyze_query_intent


class FakeRuntime:
    def __init__(self):
        self.context = {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


class QueryIntentTests(unittest.TestCase):
    def test_explicit_month_without_year_requires_clarification(self):
        intent, ambiguity = analyze_query_intent("1月份每天的销售额")

        self.assertEqual(["GMV"], intent["metrics"])
        self.assertEqual(1, intent["time"]["month"])
        self.assertEqual("day", intent["time"]["grain"])
        self.assertTrue(ambiguity["needs_clarification"])
        self.assertEqual("MISSING_YEAR_FOR_MONTH", ambiguity["code"])
        self.assertEqual(["time.year"], ambiguity["missing_slots"])
        self.assertIn("哪一年的1月", ambiguity["question"])

    def test_explicit_year_or_relative_month_is_clear(self):
        explicit_intent, explicit = analyze_query_intent("2025年1月份每天的销售额")
        relative_intent, relative = analyze_query_intent("上个月的销售额")

        self.assertFalse(explicit["needs_clarification"])
        self.assertEqual(2025, explicit_intent["time"]["year"])
        self.assertFalse(relative["needs_clarification"])
        self.assertEqual("last_month", relative_intent["time"]["relative"])

    def test_vague_metric_and_time_are_both_reported(self):
        _, ambiguity = analyze_query_intent("最近销售情况怎么样")

        self.assertEqual("MULTIPLE_AMBIGUITIES", ambiguity["code"])
        self.assertEqual(
            {"time.range", "metric"},
            set(ambiguity["missing_slots"]),
        )

    def test_vague_top_k_is_clarified_but_exact_top_k_is_kept(self):
        _, vague = analyze_query_intent("销量排名靠前的商品")
        exact_intent, exact = analyze_query_intent("GMV最高的5个商品")

        self.assertEqual("MISSING_TOP_K", vague["code"])
        self.assertFalse(exact["needs_clarification"])
        self.assertEqual(5, exact_intent["top_k"])

    def test_clear_custom_metric_is_not_treated_as_missing(self):
        average_intent, average = analyze_query_intent("订单的平均购买数量是多少")
        amount_intent, amount = analyze_query_intent("订单金额最高的5笔订单")

        self.assertIn("AVG_PURCHASE_QUANTITY", average_intent["metrics"])
        self.assertFalse(average["needs_clarification"])
        self.assertIn("ORDER_AMOUNT", amount_intent["metrics"])
        self.assertFalse(amount["needs_clarification"])

    def test_context_reference_needs_history(self):
        _, without_history = analyze_query_intent("这个指标按地区统计")
        _, with_history = analyze_query_intent(
            "这个指标按地区统计",
            has_history=True,
        )

        self.assertIn("MISSING_CONTEXT", without_history["codes"])
        self.assertNotIn("MISSING_CONTEXT", with_history["codes"])


class AmbiguityGuardNodeTests(unittest.TestCase):
    def test_guard_emits_structured_clarification_and_stops(self):
        runtime = FakeRuntime()
        update = ambiguity_guard(
            {"query": "2月份食品饮料类商品的销售额", "messages": []},
            runtime,
        )

        self.assertTrue(update["clarification_required"])
        self.assertEqual("end", route_after_ambiguity_guard(update))
        event = next(
            event
            for event in runtime.events
            if event["type"] == "clarification_required"
        )
        self.assertEqual("MISSING_YEAR_FOR_MONTH", event["code"])
        self.assertEqual(["time.year"], event["missing_slots"])

    def test_clear_query_continues_to_retrieval(self):
        runtime = FakeRuntime()
        update = ambiguity_guard(
            {"query": "2025年各地区的GMV", "messages": []},
            runtime,
        )

        self.assertFalse(update["clarification_required"])
        self.assertEqual("extract_keywords", route_after_ambiguity_guard(update))


if __name__ == "__main__":
    unittest.main()
