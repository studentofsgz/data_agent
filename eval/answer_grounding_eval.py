"""Fast deterministic evaluation for numeric answer grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.agent.answer_grounding import validate_numeric_grounding
from app.conf.app_config import app_config


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        verification = validate_numeric_grounding(
            texts=[case["answer"]],
            rows=case["rows"],
            query=case["query"],
            row_count=case["row_count"],
            tolerance=app_config.answer_generation.numeric_tolerance,
        )
        passed = verification["passed"] == case["expect_passed"]
        if case.get("expect_invalid_numbers") is not None:
            passed = passed and (
                verification["invalid_numbers"] == case["expect_invalid_numbers"]
            )
        results.append({
            "id": case["id"],
            "expected_passed": case["expect_passed"],
            "verification": verification,
            "passed": passed,
        })
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total * 100, 1) if total else 0.0,
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Evaluate grounded-answer numeric verification.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "data" / "answer_grounding_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "reports" / "smoke" / "answer_grounding_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = evaluate_cases(cases)
    summary = report["summary"]
    print("=" * 60)
    print("Grounded Answer Evaluation")
    print("=" * 60)
    for result in report["results"]:
        verification = result["verification"]
        marker = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{marker}] {result['id']} "
            f"grounded={verification['passed']} "
            f"invalid={verification['invalid_numbers']}"
        )
    print(
        f"Result: {summary['passed']}/{summary['total']} "
        f"accuracy={summary['accuracy']}%"
    )
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
