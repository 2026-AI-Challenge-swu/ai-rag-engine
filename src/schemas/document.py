from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    source: str
    category: str
    page: int
    text: str
