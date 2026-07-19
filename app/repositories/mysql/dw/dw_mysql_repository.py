import asyncio
import json
import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.conf.app_config import app_config


class SQLExecutionTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class SQLExecutionOutcome:
    rows: list[dict]
    elapsed_seconds: float
    returned_rows: int
    truncated: bool
    timeout_seconds: float
    max_result_rows: int


class DWMySQLRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_column_types(self, table_name: str) -> dict[str, str]:
        sql = f"show columns from {table_name}"
        result = await self.session.execute(text(sql))
        return {row.Field: row.Type for row in result.fetchall()}

    async def get_column_values(self, table_name: str, column_name: str, limit: int):
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        result = await self.session.execute(text(sql))
        return result.scalars().fetchall()

    async def get_db_info(self):
        result = await self.session.execute(text("select version()"))
        version = result.scalar()

        dialect = self.session.get_bind().dialect.name

        return {'version': version, 'dialect': dialect}

    async def explain_sql(self, sql):
        result = await self.session.execute(text(f"EXPLAIN FORMAT=JSON {sql}"))
        plan = result.scalar_one()
        if isinstance(plan, bytes):
            plan = plan.decode("utf-8", errors="replace")
        if isinstance(plan, str):
            plan = json.loads(plan)
        if not isinstance(plan, dict):
            raise ValueError("数据库返回的EXPLAIN结果不是JSON对象")
        return plan

    async def validate_sql(self, sql):
        return await self.explain_sql(sql)

    async def rollback(self):
        await self.session.rollback()

    async def execute_sql_sandboxed(
        self,
        sql,
        *,
        timeout_seconds: float,
        max_result_rows: int,
    ) -> SQLExecutionOutcome:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.session.execute(text(sql)),
                timeout=timeout_seconds,
            )
            fetched = result.mappings().fetchmany(max_result_rows + 1)
            truncated = len(fetched) > max_result_rows
            rows = [dict(row) for row in fetched[:max_result_rows]]
            # Every query gets a fresh transaction boundary; SELECT data is already buffered.
            await self.session.rollback()
            return SQLExecutionOutcome(
                rows=rows,
                elapsed_seconds=round(time.perf_counter() - started, 6),
                returned_rows=len(rows),
                truncated=truncated,
                timeout_seconds=timeout_seconds,
                max_result_rows=max_result_rows,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            await self.session.rollback()
            raise SQLExecutionTimeoutError(
                f"SQL执行超过{timeout_seconds}秒，已取消并回滚当前事务"
            ) from exc
        except Exception:
            await self.session.rollback()
            raise

    async def execute_sql(self, sql, timeout_seconds=None):
        cfg = app_config.sql_execution
        outcome = await self.execute_sql_sandboxed(
            sql,
            timeout_seconds=timeout_seconds or cfg.timeout_seconds,
            max_result_rows=cfg.max_result_rows,
        )
        return outcome.rows
