"""Offline evaluation runner for the Text2SQL agent.

The runner executes the existing LangGraph workflow against a fixed case set,
captures generated SQL / result events from the custom stream, and writes a
regression report that can be compared across prompt, retrieval, or graph
changes.
"""

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

from eval.metrics import aggregate_results, evaluate_case


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        cases: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("evaluation cases must be a JSON array or JSONL records")
    return data


def load_runtime_deps() -> dict[str, Any]:
    # Importing the graph loads node modules and logging sinks, so keep it lazy.
    from app.agent.context import DataAgentContext
    from app.agent.graph import graph
    from app.agent.state import DataAgentState
    from app.clients.embedding_client_manager import embedding_client_manager
    from app.clients.es_client_manager import es_client_manager
    from app.clients.mysql_client_manager import (
        dw_mysql_client_manager,
        meta_mysql_client_manager,
    )
    from app.clients.qdrant_client_manager import qdrant_client_manager
    from app.repositories.es.value_es_repository import ValueESRepository
    from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
    from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
    from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
    from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository

    return {
        "DataAgentContext": DataAgentContext,
        "DataAgentState": DataAgentState,
        "graph": graph,
        "embedding_client_manager": embedding_client_manager,
        "qdrant_client_manager": qdrant_client_manager,
        "es_client_manager": es_client_manager,
        "meta_mysql_client_manager": meta_mysql_client_manager,
        "dw_mysql_client_manager": dw_mysql_client_manager,
        "ColumnQdrantRepository": ColumnQdrantRepository,
        "ValueESRepository": ValueESRepository,
        "MetricQdrantRepository": MetricQdrantRepository,
        "MetaMySQLRepository": MetaMySQLRepository,
        "DWMySQLRepository": DWMySQLRepository,
    }


def init_clients(deps: dict[str, Any]) -> None:
    deps["embedding_client_manager"].init()
    deps["qdrant_client_manager"].init()
    deps["es_client_manager"].init()
    deps["meta_mysql_client_manager"].init()
    deps["dw_mysql_client_manager"].init()


async def close_clients(deps: dict[str, Any]) -> None:
    await deps["qdrant_client_manager"].close()
    await deps["es_client_manager"].close()
    await deps["meta_mysql_client_manager"].close()
    await deps["dw_mysql_client_manager"].close()


async def run_one_case(
    case: dict[str, Any],
    context: Any,
    graph: Any,
    state_cls: Any,
) -> dict[str, Any]:
    state = state_cls(query=case["question"], messages=case.get("messages", []))
    start = time.perf_counter()

    events: list[dict[str, Any]] = []
    sql = ""
    rows: list[dict[str, Any]] = []
    error = ""
    correction_attempts = 0
    result_received = False

    try:
        async for chunk in graph.astream(
            input=state,
            context=context,
            stream_mode="custom",
        ):
            events.append(chunk)

            event_type = chunk.get("type")
            if event_type == "sql_preview":
                sql = chunk.get("sql") or sql
            elif event_type == "result":
                data = chunk.get("data") or []
                rows = data if isinstance(data, list) else []
                result_received = True
            elif event_type == "error":
                error = chunk.get("message") or error

            step = str(chunk.get("step") or "")
            if step.startswith("校正SQL") and chunk.get("status") == "running":
                correction_attempts += 1
    except Exception as exc:  # pragma: no cover - integration path
        error = str(exc)

    elapsed = round(time.perf_counter() - start, 3)
    return evaluate_case(
        case=case,
        sql=sql,
        rows=rows,
        error=error,
        elapsed_seconds=elapsed,
        correction_attempts=correction_attempts,
        event_count=len(events),
        result_received=result_received,
    )


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    deps = load_runtime_deps()
    init_clients(deps)
    try:
        results: list[dict[str, Any]] = []
        async with (
            deps["meta_mysql_client_manager"].session_factory() as meta_session,
            deps["dw_mysql_client_manager"].session_factory() as dw_session,
        ):
            context = deps["DataAgentContext"](
                embedding_client=deps["embedding_client_manager"].client,
                column_qdrant_repository=deps["ColumnQdrantRepository"](
                    deps["qdrant_client_manager"].client
                ),
                value_es_repository=deps["ValueESRepository"](deps["es_client_manager"].client),
                metric_qdrant_repository=deps["MetricQdrantRepository"](
                    deps["qdrant_client_manager"].client
                ),
                meta_mysql_repository=deps["MetaMySQLRepository"](meta_session),
                dw_mysql_repository=deps["DWMySQLRepository"](dw_session),
            )

            for index, case in enumerate(cases, start=1):
                result = await run_one_case(
                    case,
                    context,
                    deps["graph"],
                    deps["DataAgentState"],
                )
                results.append(result)

                print(
                    f"[{index}/{len(cases)}] {case['id']} {case['question']}\n"
                    f"  sql={result['sql_generated']} executable={result['sql_executable']} "
                    f"tables={result['expected_tables_hit']} non_empty={result['not_empty_ok']} "
                    f"repair={result['correction_attempts']} elapsed={result['elapsed_seconds']}s"
                )
                if result["error"]:
                    print(f"  error={result['error'][:200]}")

        report = aggregate_results(results)
        report["details"] = results
        return report
    finally:
        await close_clients(deps)


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n" + "=" * 60)
    print("Text2SQL Offline Evaluation")
    print("=" * 60)
    print(f"Total cases: {summary['total']}")
    print(f"SQL generated: {summary['sql_generated']['rate']}%")
    print(f"SQL executable: {summary['sql_executable']['rate']}%")
    print(f"Expected table hit: {summary['expected_tables_hit']['rate']}%")
    print(f"Non-empty check: {summary['not_empty_ok']['rate']}%")
    print(f"Expected result check: {summary['expected_result_ok']['rate']}%")
    print(f"Self-repair cases: {summary['self_repair_cases']}")
    print(f"Average latency: {summary['avg_seconds']}s")

    print("\nBy difficulty:")
    for key, item in summary["by_difficulty"].items():
        print(
            f"  {key:8s} total={item['total']:3d} "
            f"exec={item['sql_executable']['rate']:5.1f}% "
            f"table={item['expected_tables_hit']['rate']:5.1f}% "
            f"avg={item['avg_seconds']}s"
        )

    print("\nBy category:")
    for key, item in summary["by_category"].items():
        print(
            f"  {key:12s} total={item['total']:3d} "
            f"exec={item['sql_executable']['rate']:5.1f}% "
            f"table={item['expected_tables_hit']['rate']:5.1f}% "
            f"avg={item['avg_seconds']}s"
        )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Run offline Text2SQL evaluation cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "questions.json",
        help="Path to JSON/JSONL evaluation cases.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "eval_report.json",
        help="Path to write evaluation report JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N cases. Useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_eval(args))
    print_summary(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()
