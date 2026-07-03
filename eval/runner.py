"""离线评测脚本：跑评测集并输出统计报告，支持按难度和类别分组统计"""
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    meta_mysql_client_manager,
    dw_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


async def run_eval(questions_path: str):
    # 初始化所有客户端
    embedding_client_manager.init()
    qdrant_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()

    with open(questions_path) as f:
        questions = json.load(f)

    results = []

    async with (
        meta_mysql_client_manager.session_factory() as meta_session,
        dw_mysql_client_manager.session_factory() as dw_session,
    ):
        context = DataAgentContext(
            embedding_client=embedding_client_manager.client,
            column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
            value_es_repository=ValueESRepository(es_client_manager.client),
            metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
            meta_mysql_repository=MetaMySQLRepository(meta_session),
            dw_mysql_repository=DWMySQLRepository(dw_session),
        )

        for i, q in enumerate(questions):
            state = DataAgentState(query=q["question"])
            start = time.time()
            sql_executable = False
            result_count = 0
            error = ""

            try:
                async for chunk in graph.astream(input=state, context=context, stream_mode="custom"):
                    if chunk.get("type") == "result":
                        result_count = len(chunk.get("data", []))
                        sql_executable = True
            except Exception as e:
                error = str(e)[:100]

            elapsed = round(time.time() - start, 2)
            results.append({
                "id": q["id"],
                "question": q["question"],
                "difficulty": q.get("difficulty", "unknown"),
                "category": q.get("category", "unknown"),
                "executable": sql_executable,
                "result_count": result_count,
                "elapsed": elapsed,
                "error": error,
            })
            print(f"[{i+1}/{len(questions)}] {q['question']}")
            print(f"   可执行={sql_executable}  行数={result_count}  耗时={elapsed}s  "
                  f"错误={'无' if not error else error}")

    # ── 统计 ──
    total = len(results)
    executable = sum(1 for r in results if r["executable"])
    avg_time = round(sum(r["elapsed"] for r in results) / total, 2)

    # 按难度分组统计
    def _group_stat(items, key):
        """按 key 对结果分组统计"""
        groups = defaultdict(lambda: {"total": 0, "executable": 0, "times": []})
        for item in items:
            g = groups[item[key]]
            g["total"] += 1
            if item["executable"]:
                g["executable"] += 1
            g["times"].append(item["elapsed"])
        return {
            k: {
                "total": v["total"],
                "executable": v["executable"],
                "rate": round(v["executable"] / v["total"] * 100, 1) if v["total"] else 0,
                "avg_seconds": round(sum(v["times"]) / len(v["times"]), 2),
            }
            for k, v in sorted(groups.items())
        }

    by_difficulty = _group_stat(results, "difficulty")
    by_category = _group_stat(results, "category")

    # 终端打印
    print(f"\n{'='*50}")
    print(f"  评测报告")
    print(f"{'='*50}")
    print(f"  总问题数: {total}")
    print(f"  SQL可执行率: {executable}/{total} ({round(executable/total*100, 1)}%)")
    print(f"  平均耗时: {avg_time}s")
    print(f"\n  ── 按难度 ──")
    for level in ["easy", "medium", "hard", "unknown"]:
        if level in by_difficulty:
            d = by_difficulty[level]
            print(f"    {level:8s}  {d['executable']}/{d['total']}  "
                  f"({d['rate']}%)  avg={d['avg_seconds']}s")
    print(f"\n  ── 按类别 ──")
    for cat, d in by_category.items():
        print(f"    {cat:12s}  {d['executable']}/{d['total']}  "
              f"({d['rate']}%)  avg={d['avg_seconds']}s")
    print(f"{'='*50}")

    # 写报告文件
    report = {
        "summary": {
            "total": total,
            "executable": executable,
            "rate": round(executable / total * 100, 1) if total else 0,
            "avg_seconds": avg_time,
            "by_difficulty": by_difficulty,
            "by_category": by_category,
        },
        "details": results,
    }
    report_path = Path(questions_path).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"详细报告: {report_path}")

    await qdrant_client_manager.close()
    await es_client_manager.close()
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()


if __name__ == "__main__":
    questions_path = Path(__file__).parent / "questions.json"
    asyncio.run(run_eval(str(questions_path)))
