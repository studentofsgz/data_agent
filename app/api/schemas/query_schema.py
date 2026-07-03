from typing import Optional

from pydantic import BaseModel


class QuerySchema(BaseModel):
    query: str
    messages: Optional[list[dict]] = None
