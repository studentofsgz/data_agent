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
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.llm_tracking import LLMCallTracker
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


def load_goldens(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        goldens = {str(item["id"]): item for item in data}
    elif isinstance(data, dict):
        goldens = {str(case_id): item for case_id, item in data.items()}
    else:
        raise ValueError("golden results must be a JSON object or array")

    for case_id, golden in goldens.items():
        if not isinstance(golden, dict):
            raise ValueError(f"golden result for {case_id} must be an object")
    return goldens


def merge_goldens(
    cases: list[dict[str, Any]],
    goldens: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {**case, **goldens.get(str(case.get("id")), {})}
        for case in cases
    ]


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
    state = state_cls(
        query=case["question"],
        messages=case.get("messages", []),
        access_context=case.get("access_context") or {},
        access_policy_result={},
        authorization_result={},
        schema_catalog=[],
    )
    start = time.perf_counter()

    events: list[dict[str, Any]] = []
    sql = ""
    rows: list[dict[str, Any]] = []
    error = ""
    correction_attempts = 0
    result_received = False
    node_timings: list[dict[str, Any]] = []
    sql_cache_status: str | None = None
    repair_guard_events: list[dict[str, Any]] = []
    schema_linking_events: list[dict[str, Any]] = []
    query_plan_events: list[dict[str, Any]] = []
    sql_sandbox_events: list[dict[str, Any]] = []
    query_intent_event: dict[str, Any] | None = None
    clarification_event: dict[str, Any] | None = None
    context_resolution_event: dict[str, Any] | None = None
    conversation_memory_event: dict[str, Any] | None = None
    confidence_events: list[dict[str, Any]] = []
    grounded_answer_event: dict[str, Any] | None = None
    access_policy_events: list[dict[str, Any]] = []
    sql_authorization_events: list[dict[str, Any]] = []
    llm_tracker = LLMCallTracker()

    try:
        async for chunk in graph.astream(
            input=state,
            context=context,
            config={
                "callbacks": [llm_tracker],
                "configurable": {
                    "thread_id": f"eval-{case.get('id')}-{uuid4().hex}",
                },
            },
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
            elif (
                event_type == "node_timing"
                and chunk.get("status") in {"success", "error"}
            ):
                node_timings.append(
                    {
                        "node": chunk.get("node"),
                        "invocation_id": chunk.get("invocation_id"),
                        "status": chunk.get("status"),
                        "elapsed_seconds": chunk.get("elapsed_seconds"),
                        "error_type": chunk.get("error_type"),
                        "error": chunk.get("error"),
                    }
                )
            elif event_type == "sql_cache":
                sql_cache_status = str(chunk.get("status") or "") or None
            elif event_type == "sql_repair_guard":
                repair_guard_events.append({
                    "status": chunk.get("status"),
                    "code": chunk.get("code"),
                    "message": chunk.get("message"),
                    "violations": chunk.get("violations") or [],
                    "attempt": chunk.get("attempt"),
                })
            elif event_type == "schema_linking":
                schema_linking_events.append({
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                })
            elif event_type == "query_plan_guard":
                query_plan_events.append({
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                })
            elif event_type == "sql_sandbox":
                sql_sandbox_events.append({
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                })
            elif event_type == "query_intent":
                query_intent_event = {
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                }
            elif event_type == "clarification_required":
                clarification_event = {
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                }
            elif event_type == "context_resolution":
                context_resolution_event = {
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                }
            elif event_type == "conversation_memory_saved":
                conversation_memory_event = {
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                }
            elif str(event_type).startswith("confidence_"):
                confidence_events.append(dict(chunk))
            elif event_type == "grounded_answer":
                grounded_answer_event = dict(chunk)
            elif event_type == "access_policy":
                access_policy_events.append({
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                })
            elif event_type == "sql_authorization":
                sql_authorization_events.append({
                    key: value
                    for key, value in chunk.items()
                    if key != "type"
                })
                if chunk.get("status") == "passed" and chunk.get("sql"):
                    sql = str(chunk["sql"])

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
        node_timings=node_timings,
        llm_calls=llm_tracker.snapshot(),
        sql_cache_status=sql_cache_status,
        repair_guard_events=repair_guard_events,
        schema_linking_events=schema_linking_events,
        query_plan_events=query_plan_events,
        sql_sandbox_events=sql_sandbox_events,
        query_intent_event=query_intent_event,
        clarification_event=clarification_event,
        context_resolution_event=context_resolution_event,
        conversation_memory_event=conversation_memory_event,
        confidence_events=confidence_events,
        grounded_answer_event=grounded_answer_event,
        access_policy_events=access_policy_events,
        sql_authorization_events=sql_authorization_events,
    )


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    goldens = load_goldens(args.goldens)
    cases = merge_goldens(cases, goldens)
    if args.only_gold:
        cases = [case for case in cases if case.get("expected_result") is not None]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("no evaluation cases selected")
    default_access_context = {
        "principal_id": args.principal_id,
        "role": args.access_role,
        "region_scope": args.region_scope,
        "source": "offline_eval",
    }
    cases = [
        {
            **case,
            "access_context": {
                **default_access_context,
                **(case.get("access_context") or {}),
                "source": "offline_eval",
            },
        }
        for case in cases
    ]

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

                sql_rule_text = ""
                if result["expected_sql_rule_ok"] is not None:
                    sql_rule_text = f" sql_rule={result['expected_sql_rule_ok']}"

                expected_result_text = ""
                if result["expected_result_ok"] is not None:
                    expected_result_text = (
                        f" result_check={result['expected_result_ok']}"
                    )

                print(
                    f"[{index}/{len(cases)}] {case['id']} {case['question']}\n"
                    f"  sql={result['sql_generated']} executable={result['sql_executable']} "
                    f"tables={result['expected_tables_hit']} non_empty={result['not_empty_ok']} "
                    f"{sql_rule_text}{expected_result_text} "
                    f"repair={result['correction_attempts']} "
                    f"clarification={result['clarification_required']} "
                    f"llm_calls={result['llm_call_count']} "
                    f"elapsed={result['elapsed_seconds']}s"
                )
                if result["slowest_node"]:
                    slowest = result["slowest_node"]
                    print(
                        f"  slowest_node={slowest['node']} "
                        f"elapsed={slowest['elapsed_seconds']}s "
                        f"llm_elapsed={result['llm_elapsed_seconds']}s"
                    )
                if result["error"]:
                    print(f"  error={result['error'][:200]}")
                if result["expected_result_diff"]:
                    diff = json.dumps(
                        result["expected_result_diff"],
                        ensure_ascii=False,
                    )
                    print(f"  result_diff={diff[:1000]}")

        report = aggregate_results(results)
        report["metadata"] = {
            "cases_file": str(args.cases),
            "goldens_file": str(args.goldens),
            "golden_cases": sum(
                result["expected_result_ok"] is not None
                for result in results
            ),
            "default_access_context": default_access_context,
        }
        report["details"] = results
        return report
    finally:
        await close_clients(deps)


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]

    def format_metric(metric: dict[str, Any]) -> str:
        metric_rate = metric["rate"]
        if metric_rate is None:
            return "N/A (0 configured)"
        return f"{metric_rate}% ({metric['success']}/{metric['total']})"

    print("\n" + "=" * 60)
    print("Text2SQL Offline Evaluation")
    print("=" * 60)
    print(f"Total cases: {summary['total']}")
    print(f"SQL generated: {format_metric(summary['sql_generated'])}")
    print(f"SQL executable: {format_metric(summary['sql_executable'])}")
    print(f"Expected table hit: {format_metric(summary['expected_tables_hit'])}")
    print(f"Empty/non-empty check: {format_metric(summary['not_empty_ok'])}")
    print(f"SQL rule check: {format_metric(summary['expected_sql_rule_ok'])}")
    print(f"Expected result check: {format_metric(summary['expected_result_ok'])}")
    print(f"Self-repair cases: {summary['self_repair_cases']}")
    if summary.get("repair_guard_stopped_cases"):
        print(
            "Repair guard stopped cases: "
            f"{summary['repair_guard_stopped_cases']} "
            f"reasons={summary['repair_stop_reasons']}"
        )
    print(f"Average latency: {summary['avg_seconds']}s")

    schema_linking = summary.get("schema_linking") or {}
    if schema_linking.get("observed_cases"):
        print("\nSchema Linking:")
        print(f"  Table recall: {format_metric(schema_linking['table_recall'])}")
        print(f"  Column Recall@K: {format_metric(schema_linking['column_recall_at_k'])}")
        print(f"  Final column recall: {format_metric(schema_linking['final_column_recall'])}")
        print(f"  Metric Recall@K: {format_metric(schema_linking['metric_recall_at_k'])}")
        print(f"  Final metric recall: {format_metric(schema_linking['final_metric_recall'])}")
        print(f"  JOIN key coverage: {format_metric(schema_linking['join_key_coverage'])}")
        print(f"  Avg candidates: {schema_linking['avg_candidate_counts']}")

    query_governance = summary.get("query_governance") or {}
    if query_governance.get("plan_checks"):
        print("\nQuery cost governance:")
        print(
            "  Plan guard: "
            f"passed={query_governance['plan_passed']} "
            f"rejected={query_governance['plan_rejected']} "
            f"reasons={query_governance['rejection_codes']}"
        )
        print(
            "  Estimates: "
            f"avg_rows={query_governance['avg_estimated_rows']} "
            f"max_rows={query_governance['max_estimated_rows']}"
        )
        print(
            "  Sandbox: "
            f"executions={query_governance['sandbox_executions']} "
            f"timeouts={query_governance['sandbox_timeouts']} "
            f"truncated={query_governance['sandbox_truncated']}"
        )

    clarification = summary.get("clarification") or {}
    if clarification.get("configured_cases"):
        print("\nIntent clarification:")
        print(
            "  Accuracy: "
            f"{format_metric(clarification['accuracy'])} "
            f"precision={clarification['precision']}% "
            f"recall={clarification['recall']}%"
        )
        print(
            "  Confusion: "
            f"TP={clarification['true_positive']} "
            f"TN={clarification['true_negative']} "
            f"FP={clarification['false_positive']} "
            f"FN={clarification['false_negative']}"
        )

    confidence = summary.get("confidence") or {}
    if confidence.get("assessed_cases"):
        print("\nConfidence governance:")
        print(
            "  Assessments: "
            f"total={confidence['assessed_cases']} "
            f"levels={confidence['levels']} "
            f"actions={confidence['actions']}"
        )
        print(
            "  Scores: "
            f"avg={confidence['avg_score']} "
            f"rejections={confidence['rejection_codes']}"
        )
        if confidence.get("configured_cases"):
            print(
                "  Expected action accuracy: "
                f"{format_metric(confidence['accuracy'])}"
            )

    grounded_answer = summary.get("grounded_answer") or {}
    if grounded_answer.get("answered_cases"):
        print("\nGrounded answer:")
        print(
            "  Answers: "
            f"total={grounded_answer['answered_cases']} "
            f"statuses={grounded_answer['statuses']} "
            f"fallbacks={grounded_answer['fallback_reasons']}"
        )
        print(
            "  Verification: "
            f"final_grounded={format_metric(grounded_answer['final_grounded'])} "
            f"model_output_grounded={format_metric(grounded_answer['model_output_grounded'])} "
            f"caught_ungrounded={grounded_answer['caught_ungrounded_cases']}"
        )
        if grounded_answer.get("configured_cases"):
            print(
                "  Expected answer accuracy: "
                f"{format_metric(grounded_answer['accuracy'])}"
            )

    access_governance = summary.get("access_governance") or {}
    if access_governance.get("policy_checks"):
        print("\nData access governance:")
        print(
            "  Checks: "
            f"policy={access_governance['policy_checks']} "
            f"sql_authorization={access_governance['authorization_checks']} "
            f"rejected={access_governance['rejected_checks']}"
        )
        print(
            "  Enforcement: "
            f"row_policy_queries={access_governance['row_policy_queries']} "
            f"row_policy_scopes={access_governance['row_policy_scopes']} "
            f"reasons={access_governance['rejection_codes']}"
        )
        print(f"  Roles: {access_governance['roles']}")
        if access_governance.get("configured_cases"):
            print(
                "  Expected access accuracy: "
                f"{format_metric(access_governance['accuracy'])}"
            )

    observability = summary.get("observability") or {}
    llm_summary = observability.get("llm") or {}
    if llm_summary.get("count"):
        print(
            "LLM calls: "
            f"{llm_summary['count']} "
            f"(errors={llm_summary['error']}, "
            f"avg={llm_summary['avg_seconds']}s, "
            f"P50={llm_summary['p50_seconds']}s, "
            f"P95={llm_summary['p95_seconds']}s, "
            f"max={llm_summary['max_seconds']}s)"
        )
        if llm_summary.get("usage_reported_calls"):
            print(
                "LLM tokens: "
                f"input={llm_summary['input_tokens']} "
                f"output={llm_summary['output_tokens']} "
                f"total={llm_summary['total_tokens']} "
                f"reported_calls={llm_summary['usage_reported_calls']}"
            )

    cache_summary = observability.get("sql_cache") or {}
    if cache_summary.get("observed"):
        print(
            "SQL cache: "
            f"hits={cache_summary['hits']} "
            f"misses={cache_summary['misses']} "
            f"bypassed={cache_summary['bypassed']}"
        )

    node_summary = observability.get("node_timings") or {}
    if node_summary:
        print("\nNode latency (slowest average first):")
        ordered_nodes = sorted(
            node_summary.items(),
            key=lambda item: item[1].get("avg_seconds") or 0,
            reverse=True,
        )
        for node, stats in ordered_nodes:
            print(
                f"  {node:24s} calls={stats['count']:3d} "
                f"avg={stats['avg_seconds']:8.3f}s "
                f"P50={stats['p50_seconds']:8.3f}s "
                f"P95={stats['p95_seconds']:8.3f}s "
                f"max={stats['max_seconds']:8.3f}s "
                f"errors={stats['error']}"
            )

    llm_by_node = llm_summary.get("by_node") or {}
    if llm_by_node:
        print("\nLLM latency by node:")
        ordered_llm_nodes = sorted(
            llm_by_node.items(),
            key=lambda item: item[1].get("avg_seconds") or 0,
            reverse=True,
        )
        for node, stats in ordered_llm_nodes:
            print(
                f"  {node:24s} calls={stats['count']:3d} "
                f"avg={stats['avg_seconds']:8.3f}s "
                f"P95={stats['p95_seconds']:8.3f}s "
                f"max={stats['max_seconds']:8.3f}s "
                f"errors={stats['error']}"
            )

    print("\nBy difficulty:")
    for key, item in summary["by_difficulty"].items():
        exec_rate = item["sql_executable"]["rate"]
        table_rate = item["expected_tables_hit"]["rate"]
        print(
            f"  {key:8s} total={item['total']:3d} "
            f"exec={exec_rate if exec_rate is not None else 'N/A'}% "
            f"table={table_rate if table_rate is not None else 'N/A'}% "
            f"avg={item['avg_seconds']}s"
        )

    print("\nBy category:")
    for key, item in summary["by_category"].items():
        exec_rate = item["sql_executable"]["rate"]
        table_rate = item["expected_tables_hit"]["rate"]
        print(
            f"  {key:12s} total={item['total']:3d} "
            f"exec={exec_rate if exec_rate is not None else 'N/A'}% "
            f"table={table_rate if table_rate is not None else 'N/A'}% "
            f"avg={item['avg_seconds']}s"
        )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Run offline Text2SQL evaluation cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=base_dir / "data" / "questions.json",
        help="Path to JSON/JSONL evaluation cases.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=base_dir / "reports" / "eval_report.json",
        help="Path to write evaluation report JSON.",
    )
    parser.add_argument(
        "--goldens",
        type=Path,
        default=base_dir / "data" / "golden_results.json",
        help="Path to manually reviewed SQL and expected results.",
    )
    parser.add_argument(
        "--only-gold",
        action="store_true",
        help="Only run cases that have an expected result.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N cases. Useful for smoke tests.",
    )
    parser.add_argument(
        "--principal-id",
        default="eval-runner",
        help="Access principal used by cases without an access_context.",
    )
    parser.add_argument(
        "--access-role",
        choices=("admin", "analyst", "region_manager"),
        default="admin",
        help="Access role used by cases without an access_context.",
    )
    parser.add_argument(
        "--region-scope",
        default="",
        help="Region scope for region_manager evaluation cases.",
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
