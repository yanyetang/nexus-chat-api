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


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    locale: str | None = Field(default=None, max_length=12)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSession(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
