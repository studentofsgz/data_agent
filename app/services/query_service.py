import json
import re
import uuid
from datetime import datetime, timezone

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.types import Command

from app.agent.access_control import (
    resolve_access_context,
    same_access_context,
    validate_access_context,
)
from app.agent.context import DataAgentContext
from app.agent.conversation_memory import build_turn_input
from app.agent.graph_runtime import graph_runtime
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class QueryService:
    THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

    def __init__(self,
                 embedding_client: HuggingFaceEndpointEmbeddings,
                 column_qdrant_repository: ColumnQdrantRepository,
                 value_es_repository: ValueESRepository,
                 metric_qdrant_repository: MetricQdrantRepository,
                 meta_mysql_repository: MetaMySQLRepository,
                 dw_mysql_repository: DWMySQLRepository,
                 workflow_graph=None):
        self.embedding_client = embedding_client
        self.column_qdrant_repository = column_qdrant_repository
        self.value_es_repository = value_es_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.meta_mysql_repository = meta_mysql_repository
        self.dw_mysql_repository = dw_mysql_repository
        self.workflow_graph = workflow_graph

    @staticmethod
    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    @staticmethod
    def _interrupts(snapshot) -> list:
        return [
            item
            for task in snapshot.tasks
            for item in task.interrupts
        ]

    @staticmethod
    def _session_expired(values: dict) -> bool:
        timestamps = [
            values.get("last_completed_at"),
            values.get("turn_started_at"),
        ]
        has_timestamp = any(timestamps)
        parsed: list[datetime] = []
        for timestamp in timestamps:
            if not timestamp:
                continue
            try:
                value = datetime.fromisoformat(str(timestamp))
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                parsed.append(value)
            except (TypeError, ValueError):
                continue
        if not parsed:
            return has_timestamp
        elapsed = (datetime.now(timezone.utc) - max(parsed)).total_seconds()
        return elapsed > app_config.conversation.session_ttl_seconds

    @staticmethod
    async def _delete_thread(workflow_graph, thread_id: str) -> None:
        checkpointer = getattr(workflow_graph, "checkpointer", None)
        delete = getattr(checkpointer, "adelete_thread", None)
        if delete is not None:
            await delete(thread_id)

    async def query(
        self,
        query: str | None = None,
        messages: list[dict] | None = None,
        thread_id: str | None = None,
        resume: str | None = None,
        principal_id: str | None = None,
        access_role: str | None = None,
        region_scope: str | None = None,
    ):
        workflow_id = str(thread_id or uuid.uuid4().hex)
        if not self.THREAD_ID_PATTERN.fullmatch(workflow_id):
            yield self._sse({
                "type": "error",
                "code": "INVALID_THREAD_ID",
                "message": "thread_id只能包含字母、数字、下划线或短横线，最长128位",
                "thread_id": workflow_id,
            })
            return

        config = {"configurable": {"thread_id": workflow_id}}
        workflow_graph = self.workflow_graph or graph_runtime.get_graph()
        context = DataAgentContext(
            embedding_client=self.embedding_client,
            column_qdrant_repository=self.column_qdrant_repository,
            value_es_repository=self.value_es_repository,
            metric_qdrant_repository=self.metric_qdrant_repository,
            meta_mysql_repository=self.meta_mysql_repository,
            dw_mysql_repository=self.dw_mysql_repository
        )

        snapshot = await workflow_graph.aget_state(config)
        expired = self._session_expired(snapshot.values)
        if expired:
            await self._delete_thread(workflow_graph, workflow_id)
            if resume is not None:
                yield self._sse({
                    "type": "error",
                    "code": "SESSION_EXPIRED",
                    "message": "会话已经过期，请重新提交完整问题",
                    "thread_id": workflow_id,
                })
                return
            snapshot = await workflow_graph.aget_state(config)

        provided_access = any(
            value is not None
            for value in (principal_id, access_role, region_scope)
        )
        existing_access = dict(snapshot.values.get("access_context") or {})
        if existing_access:
            candidate_access = resolve_access_context(
                principal_id=(
                    principal_id
                    if principal_id is not None
                    else existing_access.get("principal_id")
                ),
                role=(
                    access_role
                    if access_role is not None
                    else existing_access.get("role")
                ),
                region_scope=(
                    region_scope
                    if region_scope is not None
                    else existing_access.get("region_scope")
                ),
                source="request_demo" if provided_access else "checkpoint",
            )
            validation_error = validate_access_context(candidate_access)
            if validation_error:
                code, message = validation_error
                yield self._sse({
                    "type": "error",
                    "code": code,
                    "message": message,
                    "thread_id": workflow_id,
                })
                return
            if provided_access and not same_access_context(
                existing_access,
                candidate_access,
            ):
                yield self._sse({
                    "type": "error",
                    "code": "ACCESS_CONTEXT_MISMATCH",
                    "message": "同一thread_id不能切换访问主体、角色或数据范围",
                    "thread_id": workflow_id,
                })
                return
            access_context = existing_access
        else:
            access_context = resolve_access_context(
                principal_id=principal_id,
                role=access_role,
                region_scope=region_scope,
                source="request_demo" if provided_access else "default",
            )
            validation_error = validate_access_context(access_context)
            if validation_error:
                code, message = validation_error
                yield self._sse({
                    "type": "error",
                    "code": code,
                    "message": message,
                    "thread_id": workflow_id,
                })
                return

        if resume is not None:
            if thread_id is None:
                yield self._sse({
                    "type": "error",
                    "code": "THREAD_ID_REQUIRED_FOR_RESUME",
                    "message": "恢复任务必须提供原来的thread_id",
                    "thread_id": workflow_id,
                })
                return
            pending_interrupts = self._interrupts(snapshot)
            if not pending_interrupts:
                yield self._sse({
                    "type": "error",
                    "code": "RESUME_NOT_AVAILABLE",
                    "message": "该thread_id没有等待恢复的澄清任务",
                    "thread_id": workflow_id,
                })
                return
            graph_input = Command(resume=resume)
            yield self._sse({
                "type": "workflow_resuming",
                "thread_id": workflow_id,
            })
        else:
            if not query or not str(query).strip():
                yield self._sse({
                    "type": "error",
                    "code": "QUERY_REQUIRED",
                    "message": "新任务必须提供query",
                    "thread_id": workflow_id,
                })
                return
            if self._interrupts(snapshot):
                yield self._sse({
                    "type": "error",
                    "code": "WORKFLOW_PAUSED",
                    "message": "该会话正在等待澄清，请使用resume回答或取消",
                    "thread_id": workflow_id,
                })
                return
            mode = (
                "follow_up"
                if snapshot.values.get("last_query_intent")
                else "new"
            )
            graph_input = DataAgentState(**build_turn_input(
                query,
                messages,
                access_context,
            ))
            yield self._sse({
                "type": "workflow_started",
                "thread_id": workflow_id,
                "mode": mode,
                "principal_id": access_context["principal_id"],
                "access_role": access_context["role"],
            })

        try:
            async for chunk in workflow_graph.astream(
                input=graph_input,
                context=context,
                config=config,
                stream_mode="custom",
            ):
                yield self._sse({**chunk, "thread_id": workflow_id})

            snapshot = await workflow_graph.aget_state(config)
            interrupts = self._interrupts(snapshot)
            if interrupts:
                yield self._sse({
                    "type": "workflow_paused",
                    "thread_id": workflow_id,
                    "interrupts": [
                        {"id": item.id, "value": item.value}
                        for item in interrupts
                    ],
                })
            else:
                access_policy_result = (
                    snapshot.values.get("access_policy_result") or {}
                )
                authorization_result = (
                    snapshot.values.get("authorization_result") or {}
                )
                confidence_result = snapshot.values.get("confidence_result") or {}
                access_rejection = next(
                    (
                        result
                        for result in (access_policy_result, authorization_result)
                        if result and not result.get("passed")
                    ),
                    None,
                )
                if access_rejection:
                    yield self._sse({
                        "type": "workflow_rejected",
                        "thread_id": workflow_id,
                        "stage": "access_control",
                        "code": access_rejection.get("code"),
                        "message": access_rejection.get("message")
                        or "数据访问权限不足",
                    })
                elif confidence_result.get("action") == "reject":
                    reasons = confidence_result.get("reasons") or []
                    yield self._sse({
                        "type": "workflow_rejected",
                        "thread_id": workflow_id,
                        "stage": "confidence_guard",
                        "code": confidence_result.get("code"),
                        "message": confidence_result.get("message")
                        or "；".join(reasons)
                        or "查询证据不足，已停止SQL生成",
                        "score": confidence_result.get("score"),
                        "reasons": reasons,
                    })
                else:
                    yield self._sse({
                        "type": "workflow_completed",
                        "thread_id": workflow_id,
                        "conversation_turn": snapshot.values.get("conversation_turn", 0),
                    })
        except Exception as e:
            yield self._sse({
                "type": "error",
                "message": str(e),
                "thread_id": workflow_id,
            })
