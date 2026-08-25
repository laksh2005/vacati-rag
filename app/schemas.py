"""Request and response models. This file is the contract a client codes against."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000, examples=["What is the cancellation policy for Villa Aurora?"])
    top_k: int = Field(default=5, ge=1, le=10, description="How many source chunks to ground the answer in.")


class Citation(BaseModel):
    chunk_id: str
    document: str = Field(description="Source file the chunk came from.")
    title: str
    section: str = Field(description="Heading breadcrumb, e.g. 'Cancellation Policy > Villa Aurora'.")
    snippet: str
    score: float = Field(description="Reranker relevance score, 0-1.")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    cached: bool = False


class QueryResponse(BaseModel):
    answer: str
    # Clients should branch on this, not on the HTTP status code.
    answer_type: Literal["grounded", "insufficient_context"]
    citations: list[Citation]
    usage: Usage
    latency_ms: int


class DocumentInfo(BaseModel):
    document: str
    title: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embed_model: str
    chat_model: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class GroundedAnswer(BaseModel):
    """The JSON schema the answer model is forced to fill in."""

    answer: str
    citations: list[int] = Field(description="Numbers of the context blocks the answer is based on.")
    sufficient_context: bool


class RerankScore(BaseModel):
    """One row of the reranker's JSON output."""

    id: int = Field(description="Number of the candidate being scored.")
    score: float = Field(ge=0, le=1)
