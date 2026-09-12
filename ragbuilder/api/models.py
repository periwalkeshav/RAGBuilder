"""Pydantic request and response models.

These are the contract, and FastAPI turns them into the OpenAPI schema at
``/docs`` for free. Validation happens before any handler runs, so a malformed
request never reaches the retriever.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000, examples=["Wie viele Urlaubstage stehen mir zu?"]
    )
    session_id: str = Field("default", max_length=128, description="Groups turns into one conversation")
    strategy: Literal["fixed", "sentence", "semantic"] | None = Field(
        None, description="Chunking strategy to search; defaults to the configured one"
    )
    mode: Literal["dense", "sparse", "hybrid"] | None = Field(
        None, description="Retrieval mode; defaults to the configured one"
    )
    top_k: int | None = Field(None, ge=1, le=20)
    doc_ids: list[str] | None = Field(None, description="Restrict the search to these documents")
    use_memory: bool = Field(True, description="Include the last turns for reference resolution")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class SourceModel(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    citation: str
    excerpt: str
    score: float
    retriever: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    cited: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    confidence: float = Field(
        ..., description="0-1 retrieval confidence; below the threshold the API refuses"
    )
    refused: bool
    strategy: str
    retrieval_mode: str
    model: str
    language: str
    expanded_queries: list[str] = []
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    cited_source_count: int = 0


class IngestRequest(BaseModel):
    corpus: str | None = Field(None, description="Named corpus to download and ingest")
    strategies: list[Literal["fixed", "sentence", "semantic"]] | None = None
    download: bool = True
    force: bool = Field(False, description="Re-ingest even when the content hash is unchanged")
    embed: bool = Field(True, description="Generate embeddings after ingesting")


class IngestResponse(BaseModel):
    status: str
    detail: str
    documents_seen: int = 0
    documents_new: int = 0
    documents_changed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    duration_seconds: float = 0.0


class DocumentModel(BaseModel):
    doc_id: str
    title: str
    category: str | None = None
    language: str | None = None
    page_count: int = 0
    char_count: int = 0
    embedded_chunks: int = 0
    total_chunks: int = 0
    ingested_at: str | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentModel]
    total_documents: int
    total_chunks: int
    embedded_chunks: int


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: dict[str, ComponentHealth]
    config: dict[str, bool | int | float | str]
