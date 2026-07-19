from typing import Optional

from pydantic import BaseModel


class QuerySchema(BaseModel):
    query: Optional[str] = None
    messages: Optional[list[dict]] = None
    thread_id: Optional[str] = None
    resume: Optional[str] = None
