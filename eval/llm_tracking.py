"""Per-call LLM timing and token tracking for offline evaluation."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.agent.observability import current_node_name


def _token_usage(response: Any) -> dict[str, int | None]:
    candidates: list[dict[str, Any]] = []

    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage")
        if isinstance(token_usage, dict):
            candidates.append(token_usage)

    for generation_list in getattr(response, "generations", []) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                candidates.append(usage)
            response_metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(response_metadata, dict):
                token_usage = response_metadata.get("token_usage")
                if isinstance(token_usage, dict):
                    candidates.append(token_usage)

    for usage in candidates:
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        total_tokens = usage.get("total_tokens")
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = int(input_tokens) + int(output_tokens)
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
            return {
                "input_tokens": int(input_tokens) if input_tokens is not None else None,
                "output_tokens": int(output_tokens) if output_tokens is not None else None,
                "total_tokens": int(total_tokens) if total_tokens is not None else None,
            }

    return {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def _output_metadata(response: Any) -> tuple[int, str | None]:
    output_chars = 0
    model_name: str | None = None
    for generation_list in getattr(response, "generations", []) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            output_chars += len(str(getattr(message, "content", "") or ""))
            metadata = getattr(message, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                model_name = (
                    metadata.get("model_name")
                    or metadata.get("model")
                    or model_name
                )
    return output_chars, model_name


class LLMCallTracker(BaseCallbackHandler):
    """Collect completion-order LLM calls without storing prompt contents."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._starts: dict[str, dict[str, Any]] = {}
        self._calls: list[dict[str, Any]] = []
        self._sequence = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, tags, kwargs
        metadata = metadata or {}
        node = metadata.get("langgraph_node") or current_node_name.get() or "unknown"
        input_chars = sum(
            len(str(getattr(message, "content", "") or ""))
            for message_list in messages
            for message in message_list
        )
        serialized_kwargs = serialized.get("kwargs") or {}
        model_name = (
            metadata.get("ls_model_name")
            or serialized_kwargs.get("model_name")
            or serialized_kwargs.get("model")
            or serialized.get("name")
        )

        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            self._starts[str(run_id)] = {
                "sequence": sequence,
                "node": str(node),
                "model": str(model_name) if model_name is not None else None,
                "input_chars": input_chars,
                "started": time.perf_counter(),
            }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        usage = _token_usage(response)
        output_chars, response_model = _output_metadata(response)
        self._finish(
            run_id=run_id,
            status="success",
            usage=usage,
            output_chars=output_chars,
            response_model=response_model,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, kwargs
        self._finish(
            run_id=run_id,
            status="error",
            usage={
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
            output_chars=0,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        )

    def _finish(
        self,
        *,
        run_id: UUID,
        status: str,
        usage: dict[str, int | None],
        output_chars: int,
        response_model: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            start = self._starts.pop(str(run_id), None)
            if start is None:
                start = {
                    "sequence": self._sequence,
                    "node": current_node_name.get() or "unknown",
                    "model": None,
                    "input_chars": 0,
                    "started": time.perf_counter(),
                }
                self._sequence += 1

            self._calls.append(
                {
                    "sequence": start["sequence"],
                    "node": start["node"],
                    "model": response_model or start["model"],
                    "status": status,
                    "elapsed_seconds": round(
                        time.perf_counter() - start["started"],
                        6,
                    ),
                    "input_chars": start["input_chars"],
                    "output_chars": output_chars,
                    **usage,
                    "usage_reported": any(
                        value is not None for value in usage.values()
                    ),
                    "first_token_seconds": None,
                    "error_type": error_type,
                    "error": error_message,
                }
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(call)
                for call in sorted(self._calls, key=lambda item: item["sequence"])
            ]
