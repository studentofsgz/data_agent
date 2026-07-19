import asyncio
import unittest

from app.agent.schema_catalog import lexical_column_matches, lexical_metric_matches
from app.agent.nodes.rerank import rerank
from app.entities.column_info import ColumnInfo


class SchemaCatalogTests(unittest.TestCase):
    def test_column_aliases_add_deterministic_recall_candidates(self):
        matches = lexical_column_matches("统计各地区的销售额")
        ids = {match["id"] for match in matches}

        self.assertIn("dim_region.region_name", ids)
        self.assertIn("fact_order.order_amount", ids)

    def test_metric_aliases_cover_gmv_aov_and_sales_quantity(self):
        gmv = {match["id"] for match in lexical_metric_matches("各地区销售额")}
        aov = {match["id"] for match in lexical_metric_matches("客户客单价")}
        quantity = {match["id"] for match in lexical_metric_matches("每月总销量")}

        self.assertIn("GMV", gmv)
        self.assertIn("AOV", aov)
        self.assertIn("SALES_QUANTITY", quantity)

    def test_extended_keywords_can_match_time_columns(self):
        matches = lexical_column_matches("统计趋势", ["年份", "月份"])
        ids = {match["id"] for match in matches}

        self.assertIn("dim_date.year", ids)
        self.assertIn("dim_date.month", ids)

    def test_exact_alias_candidate_survives_low_vector_score(self):
        class Embeddings:
            async def aembed_query(self, text):
                del text
                return [1.0, 0.0]

            async def aembed_documents(self, texts):
                return [[0.0, 1.0] for _ in texts]

        class Runtime:
            def __init__(self):
                self.context = {"embedding_client": Embeddings()}
                self.events = []

            def stream_writer(self, event):
                self.events.append(event)

        column = ColumnInfo(
            id="fact_order.order_amount",
            name="order_amount",
            type="decimal",
            role="measure",
            examples=[],
            description="订单金额",
            alias=["销售额"],
            table_id="fact_order",
        )
        runtime = Runtime()
        update = asyncio.run(rerank(
            {
                "query": "销售额",
                "retrieved_columns": [column],
                "retrieved_metrics": [],
                "column_recall_sources": {column.id: ["exact_alias"]},
                "metric_recall_sources": {},
            },
            runtime,
        ))

        self.assertEqual([column.id], [item.id for item in update["retrieved_columns"]])
        event = next(event for event in runtime.events if event.get("stage") == "rerank")
        self.assertEqual(0.0, event["columns"][0]["base_score"])
        self.assertEqual(1.0, event["columns"][0]["score"])


if __name__ == "__main__":
    unittest.main()
