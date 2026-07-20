from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QuerySchema
from app.services.query_service import QueryService

query_router = APIRouter()


@query_router.post("/api/query")
async def query(
    query: QuerySchema, query_service: QueryService = Depends(get_query_service)
):
    return StreamingResponse(
        query_service.query(
            query=query.query,
            messages=query.messages,
            thread_id=query.thread_id,
            resume=query.resume,
            principal_id=query.principal_id,
            access_role=query.access_role,
            region_scope=query.region_scope,
        ),
        media_type="text/event-stream",
    )
