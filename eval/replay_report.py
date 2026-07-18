"""Replay saved SQL from an evaluation report against golden results."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mysql_client_manager import dw_mysql_client_manager
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from eval.metrics import aggregate_results, evaluate_case
from eval.runner import load_cases, load_goldens, merge_goldens, print_summary


async def replay(args: argparse.Namespace) -> dict[str, Any]:
    with args.source_report.open(encoding="utf-8") as file:
        source_report = json.load(file)

    cases = merge_goldens(load_cases(args.cases), load_goldens(args.goldens))
    cases_by_id = {
        str(case["id"]): case
        for case in cases
        if case.get("expected_result") is not None
    }
    source_details = {
        str(detail["id"]): detail
        for detail in source_report.get("details", [])
    }

    dw_mysql_client_manager.init()
    results: list[dict[str, Any]] = []
    try:
        async with dw_mysql_client_manager.session_factory() as session:
            repository = DWMySQLRepository(session)
            for case_id, case in cases_by_id.items():
                source = source_details.get(case_id)
                sql = str((source or {}).get("sql") or "")
                rows: list[dict[str, Any]] = []
                error = ""
                result_received = False
                start = time.perf_counter()

                if not source:
                    error = "case is missing from source report"
                elif not sql:
                    error = "source report has no generated SQL"
                else:
                    try:
                        rows = await repository.execute_sql(sql)
                        result_received = True
                    except Exception as exc:
                        error = str(exc)

                elapsed = round(time.perf_counter() - start, 3)
                result = evaluate_case(
                    case=case,
                    sql=sql,
                    rows=rows,
                    error=error,
                    elapsed_seconds=elapsed,
                    correction_attempts=int(
                        (source or {}).get("correction_attempts", 0)
                    ),
                    event_count=0,
                    result_received=result_received,
                )
                results.append(result)
                print(
                    f"[{case_id}] executable={result['sql_executable']} "
                    f"result_check={result['expected_result_ok']}"
                )
                if result["expected_result_diff"]:
                    print(
                        "  result_diff="
                        + json.dumps(
                            result["expected_result_diff"],
                            ensure_ascii=False,
                        )
                    )
    finally:
        await dw_mysql_client_manager.close()

    report = aggregate_results(results)
    report["metadata"] = {
        "mode": "replay_saved_sql",
        "source_report": str(args.source_report),
        "cases_file": str(args.cases),
        "goldens_file": str(args.goldens),
    }
    report["details"] = results
    return report


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Replay generated SQL from a saved report without calling the LLM."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=base_dir / "reports" / "baseline" / "metric_semantic_full.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "data" / "questions.json",
    )
    parser.add_argument(
        "--goldens",
        type=Path,
        default=base_dir / "data" / "golden_results.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "reports" / "baseline" / "golden_20_replay_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(replay(args))
    print_summary(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()
