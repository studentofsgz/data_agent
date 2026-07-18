"""Validate manually reviewed golden SQL against the current DW database."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mysql_client_manager import dw_mysql_client_manager
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from eval.metrics import compare_result_rows


async def validate_goldens(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        goldens: dict[str, dict[str, Any]] = json.load(file)

    dw_mysql_client_manager.init()
    failures = 0
    try:
        async with dw_mysql_client_manager.session_factory() as session:
            repository = DWMySQLRepository(session)
            for case_id, golden in goldens.items():
                rows = await repository.execute_sql(golden["gold_sql"])
                matched, diff = compare_result_rows(
                    rows,
                    golden["expected_result"],
                    ordered=bool(golden.get("expected_result_ordered", False)),
                    abs_tolerance=golden.get("result_abs_tolerance", 0.01),
                    column_aliases=golden.get("result_column_aliases"),
                )
                if matched:
                    print(f"[PASS] {case_id}")
                    continue

                failures += 1
                print(
                    f"[FAIL] {case_id}: "
                    f"{json.dumps(diff, ensure_ascii=False)}"
                )
    finally:
        await dw_mysql_client_manager.close()

    print(f"Validated {len(goldens)} goldens, failures={failures}")
    return failures


def main() -> None:
    path = Path(__file__).with_name("data") / "golden_results.json"
    failures = asyncio.run(validate_goldens(path))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
