from typing import Optional

from pydantic import BaseModel


class QuerySchema(BaseModel):
    query: Optional[str] = None
    messages: Optional[list[dict]] = None
    thread_id: Optional[str] = None
    resume: Optional[str] = None
    principal_id: Optional[str] = None
    access_role: Optional[str] = None
    region_scope: Optional[str] = None
