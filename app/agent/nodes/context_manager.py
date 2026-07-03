from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def context_manager(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    query = state["query"]
    messages = state.get("messages", [])

    # 无历史 → 直接透传，零 LLM 开销
    if not messages:
        return {}

    writer({"type": "progress", "step": "上下文理解", "status": "running"})

    try:
        prompt = PromptTemplate(
            template=load_prompt("context_manager"),
            input_variables=["messages", "query"]
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        rewritten = await chain.ainvoke({
            "messages": "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages[-6:]]
            ),
            "query": query
        })

        rewritten = rewritten.strip()

        # 无变化 → 标记成功但不下发步骤（避免前端空转）
        if rewritten == query:
            writer({"type": "progress", "step": "上下文理解", "status": "success"})
            return {}

        writer({"type": "progress", "step": "上下文理解", "status": "success"})
        logger.info(f"上下文改写: {query} → {rewritten}")
        return {"query": rewritten}

    except Exception as e:
        writer({"type": "progress", "step": "上下文理解", "status": "error"})
        logger.error(f"上下文改写失败: {str(e)}")
        return {}
