import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.conversation_memory import (
    is_context_dependent,
    resolve_structured_followup,
)
from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def context_manager(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    query = state["query"]
    messages = state.get("messages", [])
    previous_query = str(state.get("last_query") or "").strip()
    previous_intent = state.get("last_query_intent") or None

    resolution = resolve_structured_followup(query, previous_intent)
    if resolution["applied"]:
        writer({"type": "progress", "step": "上下文理解", "status": "running"})
        writer({"type": "context_resolution", **resolution})
        writer({"type": "progress", "step": "上下文理解", "status": "success"})
        logger.info(f"结构化上下文继承: {query} → {resolution['query_after']}")
        return {
            "query": resolution["query_after"],
            "context_resolution": resolution,
        }

    has_context = bool(messages or previous_query)
    needs_fallback = is_context_dependent(query)

    # 无上下文或当前问题已经自包含时直接透传，零LLM开销。
    if not has_context or not needs_fallback:
        return {"context_resolution": resolution}

    writer({"type": "progress", "step": "上下文理解", "status": "running"})

    try:
        prompt = PromptTemplate(
            template=load_prompt("context_manager"),
            input_variables=["messages", "query"]
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        history = list(messages[-6:])
        if not history and previous_query:
            result_summary = state.get("last_result_summary") or {}
            history = [
                {"role": "user", "content": previous_query},
                {
                    "role": "assistant",
                    "content": "上一轮结果摘要：" + json.dumps(
                        result_summary,
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ]

        rewritten = await chain.ainvoke({
            "messages": "\n".join(
                [f"{m['role']}: {m['content']}" for m in history]
            ),
            "query": query
        })

        rewritten = rewritten.strip()

        # 无变化 → 标记成功但不下发步骤（避免前端空转）
        if rewritten == query:
            unchanged = {
                **resolution,
                "strategy": "llm_fallback",
            }
            writer({"type": "context_resolution", **unchanged})
            writer({"type": "progress", "step": "上下文理解", "status": "success"})
            return {"context_resolution": unchanged}

        resolved = {
            "applied": True,
            "strategy": "llm_fallback",
            "query_before": query,
            "query_after": rewritten,
            "inherited_slots": [],
            "overridden_slots": [],
        }
        writer({"type": "context_resolution", **resolved})
        writer({"type": "progress", "step": "上下文理解", "status": "success"})
        logger.info(f"上下文改写: {query} → {rewritten}")
        return {"query": rewritten, "context_resolution": resolved}

    except Exception as e:
        failed = {**resolution, "strategy": "llm_fallback_error"}
        writer({"type": "context_resolution", **failed, "error": str(e)[:500]})
        writer({"type": "progress", "step": "上下文理解", "status": "error"})
        logger.error(f"上下文改写失败: {str(e)}")
        return {"context_resolution": failed}
