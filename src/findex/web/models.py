from pydantic import BaseModel


class SearchHit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    page_size: int
    took_ms: float
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsResponse(BaseModel):
    documents: int
    terms: int
    tokens: int
    has_positions: bool
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
