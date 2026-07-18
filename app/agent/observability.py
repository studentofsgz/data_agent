"""Lightweight observability helpers for LangGraph nodes."""

from __future__ import annotations

import inspect
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable


current_node_name: ContextVar[str | None] = ContextVar(
    "current_node_name",
    default=None,
)


def _runtime_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    runtime = kwargs.get("runtime")
    if runtime is not None:
        return runtime
    return args[1] if len(args) > 1 else None


def _emit(runtime: Any, event: dict[str, Any]) -> None:
    writer = getattr(runtime, "stream_writer", None)
    if writer is None:
        return
    try:
        writer(event)
    except Exception:
        # Observability must never change the graph's business behavior.
        pass


def instrument_node(node_name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync or async node and emit start/completion timing events."""

    @wraps(node)
    async def monitored(*args: Any, **kwargs: Any) -> Any:
        runtime = _runtime_from_call(args, kwargs)
        invocation_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = "success"
        error_type: str | None = None
        error_message: str | None = None
        token = current_node_name.set(node_name)

        _emit(
            runtime,
            {
                "type": "node_timing",
                "node": node_name,
                "invocation_id": invocation_id,
                "status": "running",
            },
        )

        try:
            result = node(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            error_message = str(exc)[:500]
            raise
        finally:
            elapsed = round(time.perf_counter() - started, 6)
            _emit(
                runtime,
                {
                    "type": "node_timing",
                    "node": node_name,
                    "invocation_id": invocation_id,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "error_type": error_type,
                    "error": error_message,
                },
            )
            current_node_name.reset(token)

    return monitored
