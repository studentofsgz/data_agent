"""Fast deterministic evaluation for the pre-SQL confidence boundary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.agent.confidence import evaluate_confidence
from app.agent.query_intent import extract_query_intent


BASE_TABLES = [
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


def _profile_inputs(case: dict[str, Any]) -> dict[str, Any]:
    profile = case["profile"]
    intent = extract_query_intent(case["question"])
    inputs: dict[str, Any] = {
        "query_intent": intent,
        "table_infos": BASE_TABLES,
        "metric_infos": [{"name": "GMV"}],
        "metric_semantics": {"metrics": [{"name": "GMV"}]},
        "column_recall_sources": {
            "fact_order.order_amount": ["exact_alias"],
            "dim_region.region_name": ["exact_alias"],
        },
        "metric_recall_sources": {"GMV": ["exact_alias"]},
        "column_candidate_scores": {
            "fact_order.order_amount": 0.82,
            "dim_region.region_name": 0.79,
        },
        "metric_candidate_scores": {"GMV": 0.84},
        "schema_linking_degraded": False,
    }

    if profile == "known_metric_exact":
        return inputs
    if profile == "detail_exact":
        return {
            **inputs,
            "table_infos": [{
                "name": "dim_region",
                "columns": [{"name": "region_name"}],
            }],
            "metric_infos": [],
            "metric_semantics": {"metrics": []},
            "column_recall_sources": {
                "dim_region.region_name": ["exact_alias"],
            },
            "metric_recall_sources": {},
            "column_candidate_scores": {"dim_region.region_name": 0.79},
            "metric_candidate_scores": {},
        }
    if profile == "degraded_metric":
        return {
            **inputs,
            "column_recall_sources": {},
            "metric_recall_sources": {},
            "column_candidate_scores": {},
            "metric_candidate_scores": {},
            "schema_linking_degraded": True,
        }
    if profile == "metric_conflict":
        return {
            **inputs,
            "metric_infos": [{"name": "GMV"}, {"name": "AOV"}],
            "column_recall_sources": {},
            "metric_recall_sources": {},
            "column_candidate_scores": {},
            "metric_candidate_scores": {},
        }
    if profile == "no_schema":
        return {
            **inputs,
            "table_infos": [],
            "column_recall_sources": {},
            "column_candidate_scores": {},
        }
    if profile == "missing_metric_column":
        return {
            **inputs,
            "table_infos": [{
                "name": "fact_order",
                "columns": [{"name": "order_amount"}],
            }],
            "metric_infos": [],
            "metric_semantics": {"metrics": []},
            "column_recall_sources": {
                "fact_order.order_amount": ["exact_alias"],
            },
            "metric_recall_sources": {},
            "column_candidate_scores": {"fact_order.order_amount": 0.82},
            "metric_candidate_scores": {},
        }
    raise ValueError(f"unknown confidence profile: {profile}")


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        actual = evaluate_confidence(**_profile_inputs(case))
        expected = {
            "level": case["expect_level"],
            "action": case["expect_action"],
            "code": case["expect_code"],
        }
        passed = all(actual[key] == value for key, value in expected.items())
        results.append({
            "id": case["id"],
            "question": case["question"],
            "profile": case["profile"],
            "expected": expected,
            "actual": actual,
            "passed": passed,
        })

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total * 100, 1) if total else 0.0,
            "levels": dict(Counter(
                result["actual"]["level"] for result in results
            )),
            "actions": dict(Counter(
                result["actual"]["action"] for result in results
            )),
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic SQL confidence policy cases.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "data" / "confidence_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "reports" / "smoke" / "confidence_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = evaluate_cases(cases)
    summary = report["summary"]

    print("=" * 60)
    print("Confidence Governance Evaluation")
    print("=" * 60)
    for result in report["results"]:
        actual = result["actual"]
        marker = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{marker}] {result['id']} "
            f"level={actual['level']} action={actual['action']} "
            f"code={actual['code']} score={actual['score']}"
        )
    print(
        f"Result: {summary['passed']}/{summary['total']} "
        f"accuracy={summary['accuracy']}%"
    )
    print(f"Levels: {summary['levels']}")
    print(f"Actions: {summary['actions']}")
    print("=" * 60)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report written to: {args.report}")
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
