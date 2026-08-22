from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str


class ContextItem(BaseModel):
    doc_id: str
    source: str
    category: str
    page: int
    text: str


class SearchResponse(BaseModel):
    context: list[ContextItem]
