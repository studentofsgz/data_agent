import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agent.context import DataAgentContext
from app.agent.nodes.add_extra_context import add_extra_context
from app.agent.nodes.ambiguity_guard import ambiguity_guard
from app.agent.nodes.context_manager import context_manager
from app.agent.nodes.clarify_intent import clarify_intent
from app.agent.nodes.confidence_guard import confidence_guard
from app.agent.nodes.confirm_confidence import confirm_confidence
from app.agent.nodes.correct_sql import correct_sql
from app.agent.nodes.execute_sql import execute_sql
from app.agent.nodes.extract_keywords import extract_keywords
from app.agent.nodes.filter_metric import filter_metric
from app.agent.nodes.filter_table import filter_table
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.merge_retrieved_info import merge_retrieved_info
from app.agent.nodes.recall_column import recall_column
from app.agent.nodes.recall_metric import recall_metric
from app.agent.nodes.recall_value import recall_value
from app.agent.nodes.remember_turn import remember_turn
from app.agent.nodes.repair_guard import repair_guard
from app.agent.nodes.rerank import rerank
from app.agent.nodes.audit_sql import audit_sql
from app.agent.nodes.query_plan_guard import query_plan_guard
from app.agent.nodes.validate_sql import validate_sql
from app.agent.observability import instrument_node
from app.agent.state import DataAgentState
from app.agent.state import MAX_SQL_RETRIES
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


def route_after_audit(state: DataAgentState) -> str:
    if state.get("error") is None:
        return "validate_sql"
    if state.get("retry_count", 0) >= MAX_SQL_RETRIES:
        return "end"
    return "correct_sql"


def route_after_validation(state: DataAgentState) -> str:
    if state.get("error") is None:
        return "query_plan_guard"
    if state.get("retry_count", 0) >= MAX_SQL_RETRIES:
        return "end"
    return "correct_sql"


def route_after_query_plan(state: DataAgentState) -> str:
    result = state.get("query_plan_result") or {}
    if state.get("error") is None and result.get("passed"):
        return "execute_sql"
    return "end"


def route_after_execution(state: DataAgentState) -> str:
    return "remember_turn" if state.get("error") is None else "end"


def route_after_repair_guard(state: DataAgentState) -> str:
    result = state.get("repair_guard_result") or {}
    if result.get("passed"):
        return "audit_sql"
    return "end"


def route_after_ambiguity_guard(state: DataAgentState) -> str:
    result = state.get("ambiguity_result") or {}
    if result.get("action") == "clarify":
        return "clarify_intent"
    if result.get("action") == "stop":
        return "end"
    return "extract_keywords"


def route_after_clarification(state: DataAgentState) -> str:
    if state.get("clarification_cancelled"):
        return "end"
    return "ambiguity_guard"


def route_after_confidence_guard(state: DataAgentState) -> str:
    action = (state.get("confidence_result") or {}).get("action")
    if action == "proceed":
        return "generate_sql"
    if action == "confirm":
        return "confirm_confidence"
    return "end"


def route_after_confidence_confirmation(state: DataAgentState) -> str:
    return "generate_sql" if state.get("confidence_confirmed") else "end"

graph_builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)
# 添加节点并统一启用耗时监控
nodes = {
    "context_manager": context_manager,
    "ambiguity_guard": ambiguity_guard,
    "clarify_intent": clarify_intent,
    "confidence_guard": confidence_guard,
    "confirm_confidence": confirm_confidence,
    "extract_keywords": extract_keywords,
    "recall_column": recall_column,
    "recall_value": recall_value,
    "recall_metric": recall_metric,
    "rerank": rerank,
    "merge_retrieved_info": merge_retrieved_info,
    "filter_metric": filter_metric,
    "filter_table": filter_table,
    "add_extra_context": add_extra_context,
    "generate_sql": generate_sql,
    "validate_sql": validate_sql,
    "audit_sql": audit_sql,
    "query_plan_guard": query_plan_guard,
    "correct_sql": correct_sql,
    "repair_guard": repair_guard,
    "execute_sql": execute_sql,
    "remember_turn": remember_turn,
}
for node_name, node in nodes.items():
    graph_builder.add_node(node_name, instrument_node(node_name, node))

# 添加关系
graph_builder.add_edge(START, "context_manager")
graph_builder.add_edge("context_manager", "ambiguity_guard")
graph_builder.add_conditional_edges(
    "ambiguity_guard",
    route_after_ambiguity_guard,
    {
        "clarify_intent": "clarify_intent",
        "extract_keywords": "extract_keywords",
        "end": END,
    },
)
graph_builder.add_conditional_edges(
    "clarify_intent",
    route_after_clarification,
    {"ambiguity_guard": "ambiguity_guard", "end": END},
)
graph_builder.add_edge("extract_keywords", "recall_column")
graph_builder.add_edge("extract_keywords", "recall_value")
graph_builder.add_edge("extract_keywords", "recall_metric")
graph_builder.add_edge("recall_column", "rerank")
graph_builder.add_edge("recall_value", "rerank")
graph_builder.add_edge("recall_metric", "rerank")
graph_builder.add_edge("rerank", "merge_retrieved_info")
graph_builder.add_edge("merge_retrieved_info", "filter_table")
graph_builder.add_edge("merge_retrieved_info", "filter_metric")
graph_builder.add_edge("filter_table", "add_extra_context")
graph_builder.add_edge("filter_metric", "add_extra_context")
graph_builder.add_edge("add_extra_context", "confidence_guard")
graph_builder.add_conditional_edges(
    "confidence_guard",
    route_after_confidence_guard,
    {
        "generate_sql": "generate_sql",
        "confirm_confidence": "confirm_confidence",
        "end": END,
    },
)
graph_builder.add_conditional_edges(
    "confirm_confidence",
    route_after_confidence_confirmation,
    {"generate_sql": "generate_sql", "end": END},
)
graph_builder.add_edge("generate_sql", "audit_sql")
graph_builder.add_conditional_edges(
    "audit_sql",
    route_after_audit,
    {"validate_sql": "validate_sql", "correct_sql": "correct_sql", "end": END},
)

graph_builder.add_conditional_edges(
    "validate_sql",
    route_after_validation,
    {"query_plan_guard": "query_plan_guard", "correct_sql": "correct_sql", "end": END},
)

graph_builder.add_conditional_edges(
    "query_plan_guard",
    route_after_query_plan,
    {"execute_sql": "execute_sql", "end": END},
)

graph_builder.add_edge("correct_sql", "repair_guard")
graph_builder.add_conditional_edges(
    "repair_guard",
    route_after_repair_guard,
    {"audit_sql": "audit_sql", "end": END},
)
graph_builder.add_conditional_edges(
    "execute_sql",
    route_after_execution,
    {"remember_turn": "remember_turn", "end": END},
)
graph_builder.add_edge("remember_turn", END)


def build_graph(checkpointer=None):
    """Compile the same workflow against an injected checkpoint backend."""
    return graph_builder.compile(checkpointer=checkpointer or InMemorySaver())


checkpointer = InMemorySaver()
graph = build_graph(checkpointer)



if __name__ == '__main__':
    async def test():
        embedding_client_manager.init()
        qdrant_client_manager.init()
        es_client_manager.init()
        meta_mysql_client_manager.init()
        dw_mysql_client_manager.init()

        async with meta_mysql_client_manager.session_factory() as meta_session, dw_mysql_client_manager.session_factory() as dw_session:
            meta_mysql_repository = MetaMySQLRepository(meta_session)
            dw_mysql_repository = DWMySQLRepository(dw_session)
            column_qdrant_repository = ColumnQdrantRepository(qdrant_client_manager.client)
            value_es_repository = ValueESRepository(es_client_manager.client)
            metric_qdrant_repository = MetricQdrantRepository(qdrant_client_manager.client)

            context = DataAgentContext(
                embedding_client=embedding_client_manager.client,
                column_qdrant_repository=column_qdrant_repository,
                value_es_repository=value_es_repository,
                metric_qdrant_repository=metric_qdrant_repository,
                meta_mysql_repository=meta_mysql_repository,
                dw_mysql_repository=dw_mysql_repository
            )
            state = DataAgentState(query="统计去年各地区的销售总额")
            async for chunk in graph.astream(
                input=state,
                context=context,
                config={"configurable": {"thread_id": "graph-main-test"}},
                stream_mode="custom",
            ):
                print(chunk)

        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()


    asyncio.run(test())
