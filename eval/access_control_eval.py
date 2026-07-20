"""Fast deterministic evaluation for schema, column and row access policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.agent.access_control import (
    apply_schema_access_policy,
    authorize_sql_text,
    resolve_access_context,
)


BASE_TABLES = [
    {
        "name": "fact_order",
        "role": "fact",
        "description": "订单事实表",
        "columns": [
            {"name": "order_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "订单ID", "alias": []},
            {"name": "customer_id", "type": "varchar", "role": "foreign_key", "examples": [], "description": "客户ID", "alias": []},
            {"name": "region_id", "type": "varchar", "role": "foreign_key", "examples": [], "description": "地区ID", "alias": []},
            {"name": "sales_amount", "type": "decimal", "role": "measure", "examples": [], "description": "销售额", "alias": []},
        ],
    },
    {
        "name": "dim_region",
        "role": "dim",
        "description": "地区维度",
        "columns": [
            {"name": "region_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "地区ID", "alias": []},
            {"name": "region_name", "type": "varchar", "role": "dimension", "examples": [], "description": "地区名称", "alias": []},
        ],
    },
    {
        "name": "dim_customer",
        "role": "dim",
        "description": "客户维度",
        "columns": [
            {"name": "customer_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "客户ID", "alias": []},
            {"name": "customer_name", "type": "varchar", "role": "dimension", "examples": [], "description": "客户姓名", "alias": []},
        ],
    },
]


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    context = resolve_access_context(**case.get("access_context", {}), source="eval")
    _, visible, policy_result = apply_schema_access_policy(
        table_infos=BASE_TABLES,
        query=case["question"],
        access_context=context,
    )
    stage = "schema_policy"
    actual = policy_result
    if policy_result["passed"] and case.get("sql"):
        stage = "sql_authorization"
        actual = authorize_sql_text(
            sql=case["sql"],
            access_context=context,
        )

    expected = case["expected"]
    checks = {
        "stage": stage == expected["stage"],
        "passed": bool(actual.get("passed")) == bool(expected["passed"]),
        "code": actual.get("code") == expected["code"],
    }
    if "row_policy_scopes" in expected:
        checks["row_policy_scopes"] = (
            actual.get("row_policy_scopes") == expected["row_policy_scopes"]
        )
    if "sql_contains" in expected:
        checks["sql_contains"] = all(
            token in str(actual.get("sql") or "")
            for token in expected["sql_contains"]
        )
    if "removed_columns" in expected:
        checks["removed_columns"] = set(expected["removed_columns"]).issubset(
            set(policy_result.get("removed_columns") or [])
        )
    if "visible_columns" in expected:
        visible_columns = {
            f"{table['name']}.{column['name']}"
            for table in visible
            for column in table.get("columns") or []
        }
        checks["visible_columns"] = set(expected["visible_columns"]).issubset(
            visible_columns
        )
    return {
        "id": case["id"],
        "question": case["question"],
        "role": context["role"],
        "expected": expected,
        "actual": actual,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_case(case) for case in cases]
    passed = sum(result["passed"] for result in results)
    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "accuracy": round(passed / len(results) * 100, 1) if results else 0.0,
            "roles": dict(Counter(result["role"] for result in results)),
            "codes": dict(Counter(
                str(result["actual"].get("code") or "UNKNOWN")
                for result in results
            )),
            "row_policy_scopes": sum(
                int(result["actual"].get("row_policy_scopes") or 0)
                for result in results
            ),
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic data access policies.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "data" / "access_control_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "reports" / "smoke" / "access_control_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = evaluate_cases(cases)
    summary = report["summary"]

    print("=" * 60)
    print("Data Access Governance Evaluation")
    print("=" * 60)
    for result in report["results"]:
        marker = "PASS" if result["passed"] else "FAIL"
        actual = result["actual"]
        print(
            f"[{marker}] {result['id']} role={result['role']} "
            f"code={actual.get('code')} passed={actual.get('passed')}"
        )
    print(
        f"Result: {summary['passed']}/{summary['total']} "
        f"accuracy={summary['accuracy']}%"
    )
    print(f"Roles: {summary['roles']}")
    print(f"Codes: {summary['codes']}")
    print(f"Row policy scopes: {summary['row_policy_scopes']}")
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
