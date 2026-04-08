from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    indexed: int
    total: int


class SearchResult(BaseModel):
    product_id: str
    chunk_text: str
    metadata: dict = Field(default_factory=dict)
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class RetrievalFilters(BaseModel):
    category: str | None = Field(default=None, min_length=1, max_length=120)
    brand: str | None = Field(default=None, min_length=1, max_length=120)
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=20)
    filters: RetrievalFilters | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    locale: str | None = Field(default=None, max_length=12)
    filters: RetrievalFilters | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSession(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class IngestJobAccepted(BaseModel):
    job_id: str
    status: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    indexed: int = 0
    total: int = 0
    error: str | None = None
