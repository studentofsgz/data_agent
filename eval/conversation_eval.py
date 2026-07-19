"""Fast deterministic evaluation for structured multi-turn context resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.agent.conversation_memory import resolve_structured_followup
from app.agent.query_intent import extract_query_intent


BASE_DIR = Path(__file__).parent


def evaluate_conversation_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        previous_intent = extract_query_intent(case["previous_question"])
        actual = resolve_structured_followup(case["question"], previous_intent)
        rewritten = actual["query_after"]
        missing = [
            item for item in case.get("expect_contains", [])
            if item not in rewritten
        ]
        unexpected = [
            item for item in case.get("expect_excludes", [])
            if item in rewritten
        ]
        passed = (
            actual["strategy"] == case["expect_strategy"]
            and actual["applied"] == case["expect_applied"]
            and not missing
            and not unexpected
        )
        results.append({
            "id": case["id"],
            "previous_question": case["previous_question"],
            "question": case["question"],
            "passed": passed,
            "expected_strategy": case["expect_strategy"],
            "actual": actual,
            "missing_fragments": missing,
            "unexpected_fragments": unexpected,
        })

    passed_count = sum(item["passed"] for item in results)
    total = len(results)
    return {
        "summary": {
            "total": total,
            "passed": passed_count,
            "accuracy": round(passed_count / total * 100, 1) if total else 0.0,
        },
        "cases": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn context resolution")
    parser.add_argument(
        "--cases",
        type=Path,
        default=BASE_DIR / "data" / "conversation_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BASE_DIR / "reports" / "smoke" / "conversation_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = evaluate_conversation_cases(cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = report["summary"]
    print("=" * 60)
    print("Conversation Memory Evaluation")
    print("=" * 60)
    print(f"Total cases: {summary['total']}")
    print(f"Passed: {summary['passed']}/{summary['total']}")
    print(f"Accuracy: {summary['accuracy']}%")
    print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()
