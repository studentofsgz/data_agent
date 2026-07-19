"""Application lifecycle manager for the durable LangGraph checkpointer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.graph import build_graph, graph as in_memory_graph
from app.conf.app_config import app_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GraphRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        persistent: bool | None = None,
    ):
        configured_path = checkpoint_path or app_config.conversation.checkpoint_path
        path = Path(configured_path)
        self.checkpoint_path = path if path.is_absolute() else PROJECT_ROOT / path
        self.persistent = (
            app_config.conversation.persistent_checkpointer
            if persistent is None
            else persistent
        )
        self._context_manager: Any = None
        self._checkpointer: Any = None
        self._graph = in_memory_graph

    async def start(self) -> None:
        if not self.persistent or self._context_manager is not None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._context_manager = AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        )
        self._checkpointer = await self._context_manager.__aenter__()
        await self._checkpointer.setup()
        self._graph = build_graph(self._checkpointer)

    async def close(self) -> None:
        if self._context_manager is None:
            return
        await self._context_manager.__aexit__(None, None, None)
        self._context_manager = None
        self._checkpointer = None
        self._graph = in_memory_graph

    def get_graph(self):
        return self._graph


graph_runtime = GraphRuntime()
