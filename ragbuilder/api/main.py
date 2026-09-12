"""FastAPI backend.

Endpoints
---------
``POST /query``         answer a question over the knowledge base
``POST /query/stream``  the same, streamed as server-sent events
``POST /ingest``        download, parse, chunk and embed a corpus (background)
``GET  /documents``     what is currently in the knowledge base
``GET  /health``        component-level health for PostgreSQL, Qdrant and Ollama
``GET  /stats``         chunk counts per strategy, vector store info, query volume

Notes on the shape of this service: the LLM and embedding calls are blocking and
CPU-bound, so they run in a thread via ``run_in_threadpool`` rather than pretending
to be async. Declaring a handler ``async def`` and then doing blocking work inside
it is the classic way to stall an event loop - the concurrency here is real.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# With PEP 563 the endpoint annotations become strings, and slowapi's
# `@limiter.limit` decorator replaces the function's __globals__ - so FastAPI
# cannot resolve `QueryRequest` when it builds the OpenAPI schema and every
# route raises PydanticUndefinedAnnotation at import time.

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ragbuilder import db
from ragbuilder.api.models import (
    ComponentHealth,
    DocumentModel,
    DocumentsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from ragbuilder.config import get_config
from ragbuilder.embeddings.store import get_store
from ragbuilder.llm.client import LLMError, get_client
from ragbuilder.rag.chain import get_chain
from ragbuilder.retrieval import sparse

LOG = logging.getLogger("ragbuilder.api")
VERSION = "1.0.0"

config = get_config()
limiter = Limiter(key_func=get_remote_address, default_limits=[config.api.rate_limit])

_ingest_state: dict[str, Any] = {"running": False, "started_at": None, "last_result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    LOG.info("RAGBuilder API %s starting", VERSION)
    try:
        db.wait_for_postgres()
        db.init_schema()
    except Exception as exc:  # noqa: BLE001 - report at /health rather than crash-loop
        LOG.error("database initialisation failed: %s", exc)
    yield
    LOG.info("RAGBuilder API shutting down")


app = FastAPI(
    title="RAGBuilder API",
    version=VERSION,
    description=(
        "Retrieval-augmented question answering over German labour law, running entirely "
        "on local models: multilingual-e5 embeddings, Qdrant hybrid search and Ollama "
        "generation. No document or question leaves the host."
    ),
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


# --------------------------------------------------------------------- query
@app.post("/query", response_model=QueryResponse, tags=["query"])
@limiter.limit(config.api.rate_limit)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Answer a question using only the ingested documents."""
    chain = get_chain()
    try:
        answer = await run_in_threadpool(
            chain.ask,
            payload.question,
            session_id=payload.session_id,
            strategy=payload.strategy,
            mode=payload.mode,
            top_k=payload.top_k,
            doc_ids=payload.doc_ids,
            use_memory=payload.use_memory,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"language model unavailable: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(**answer.to_dict())


@app.post("/query/stream", tags=["query"])
@limiter.limit(config.api.rate_limit)
async def query_stream(request: Request, payload: QueryRequest):
    """Stream the answer as server-sent events: sources first, then tokens."""
    import json
    import queue
    import threading

    chain = get_chain()
    events: "queue.Queue[str | None]" = queue.Queue()

    def produce() -> None:
        try:
            for event in chain.stream(
                payload.question,
                session_id=payload.session_id,
                strategy=payload.strategy,
                mode=payload.mode,
                top_k=payload.top_k,
            ):
                events.put(json.dumps(event, default=str))
        except Exception as exc:  # noqa: BLE001 - surface the failure to the client
            events.put(json.dumps({"type": "error", "error": str(exc)}))
        finally:
            events.put(None)

    threading.Thread(target=produce, daemon=True).start()

    async def event_source():
        while True:
            item = await run_in_threadpool(events.get)
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -------------------------------------------------------------------- ingest
def _run_ingestion(payload: IngestRequest) -> None:
    from ragbuilder.embeddings.pipeline import embed_all
    from ragbuilder.ingestion.pipeline import ingest_corpus

    _ingest_state.update({"running": True, "started_at": time.time(), "last_result": None})
    try:
        strategies = payload.strategies or [config.chunking.strategy]
        stats = ingest_corpus(
            corpus_name=payload.corpus,
            strategies=strategies,
            download=payload.download,
            force=payload.force,
        )
        if payload.embed:
            embed_all(strategies=list(strategies))
        # New documents mean the cached BM25 index is stale.
        sparse.invalidate()
        _ingest_state["last_result"] = stats.as_dict()
        LOG.info("ingestion finished: %s", stats.as_dict())
    except Exception as exc:  # noqa: BLE001
        LOG.exception("ingestion failed")
        _ingest_state["last_result"] = {"error": str(exc)}
    finally:
        _ingest_state["running"] = False


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
@limiter.limit("5/minute")
async def ingest(request: Request, payload: IngestRequest, background: BackgroundTasks) -> IngestResponse:
    """Kick off ingestion in the background.

    Ingesting 24 PDFs plus embedding takes minutes, far beyond any sane HTTP
    timeout, so this returns immediately and progress is polled via ``/stats``.
    """
    if _ingest_state["running"]:
        raise HTTPException(status_code=409, detail="an ingestion run is already in progress")

    background.add_task(_run_ingestion, payload)
    return IngestResponse(
        status="accepted",
        detail=(
            f"ingesting corpus {payload.corpus or config.corpus.name!r} with strategies "
            f"{payload.strategies or [config.chunking.strategy]}; poll GET /stats for progress"
        ),
    )


# ----------------------------------------------------------------- documents
@app.get("/documents", response_model=DocumentsResponse, tags=["knowledge base"])
async def documents() -> DocumentsResponse:
    """Everything currently in the knowledge base."""
    try:
        rows = await run_in_threadpool(db.list_documents)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc

    models = [
        DocumentModel(
            doc_id=row["doc_id"],
            title=row["title"],
            category=row.get("category"),
            language=row.get("language"),
            page_count=row.get("page_count") or 0,
            char_count=row.get("char_count") or 0,
            embedded_chunks=row.get("embedded_chunks") or 0,
            total_chunks=row.get("total_chunks") or 0,
            ingested_at=str(row.get("ingested_at")) if row.get("ingested_at") else None,
        )
        for row in rows
    ]
    return DocumentsResponse(
        documents=models,
        total_documents=len(models),
        total_chunks=sum(m.total_chunks for m in models),
        embedded_chunks=sum(m.embedded_chunks for m in models),
    )


@app.delete("/documents/{doc_id}", tags=["knowledge base"])
async def delete_document(doc_id: str) -> dict[str, str]:
    await run_in_threadpool(db.delete_document, doc_id)
    sparse.invalidate()
    return {"status": "deleted", "doc_id": doc_id}


# -------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Component-level health. Returns 200 even when degraded, on purpose:

    a load balancer should know *which* dependency is down, and a 503 that hides
    the reason makes that harder, not easier.
    """
    components: dict[str, ComponentHealth] = {}

    try:
        await run_in_threadpool(db.wait_for_postgres, 1, 0.5)
        components["postgres"] = ComponentHealth(status="ok")
    except Exception as exc:  # noqa: BLE001
        components["postgres"] = ComponentHealth(status="down", detail=str(exc)[:200])

    store = get_store()
    if await run_in_threadpool(store.healthy):
        info = await run_in_threadpool(store.info)
        components["qdrant"] = ComponentHealth(
            status="ok" if info.get("points") else "degraded",
            detail=f"{info.get('points', 0)} point(s), {info.get('dimensions', '?')} dimensions",
        )
    else:
        components["qdrant"] = ComponentHealth(
            status="down", detail=f"unreachable at {config.vector_store.url}"
        )

    client = get_client()
    if await run_in_threadpool(client.healthy):
        models = await run_in_threadpool(client.available_models)
        wanted = config.llm.model
        installed = any(m.split(":")[0] == wanted.split(":")[0] for m in models)
        components["ollama"] = ComponentHealth(
            status="ok" if installed else "degraded",
            detail=f"models: {', '.join(models) or 'none'}"
            + ("" if installed else f" (configured model {wanted!r} is not pulled)"),
        )
    else:
        components["ollama"] = ComponentHealth(status="down", detail=f"unreachable at {config.llm.host}")

    statuses = {c.status for c in components.values()}
    overall = "down" if "down" in statuses else ("degraded" if "degraded" in statuses else "ok")

    return HealthResponse(
        status=overall,
        version=VERSION,
        components=components,
        config={
            "embedding_model": config.embedding.model,
            "embedding_dimensions": config.embedding.dimensions,
            "llm_model": config.llm.model,
            "chunking_strategy": config.chunking.strategy,
            "retrieval_mode": config.retrieval.strategy,
            "top_k": config.retrieval.top_k,
            "min_confidence": config.rag.min_confidence,
            "query_expansion": config.retrieval.query_expansion,
        },
    )


@app.get("/stats", tags=["ops"])
async def stats() -> dict[str, Any]:
    """Chunk counts, vector store state, ingestion progress and query volume."""
    chunk_rows = await run_in_threadpool(db.chunk_stats)
    store_info = await run_in_threadpool(get_store().info)

    def query_counts() -> dict[str, Any]:
        with db.cursor(dict_rows=True) as cur:
            cur.execute(
                """
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE refused) AS refused,
                       round(avg(confidence)::numeric, 3) AS avg_confidence,
                       round(avg(latency_ms)) AS avg_latency_ms
                FROM query_log
                """
            )
            return dict(cur.fetchone() or {})

    try:
        queries = await run_in_threadpool(query_counts)
    except Exception:  # noqa: BLE001
        queries = {}

    return {
        "chunks": chunk_rows,
        "vector_store": store_info,
        "queries": queries,
        "ingestion": _ingest_state,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "RAGBuilder", "version": VERSION, "docs": "/docs", "health": "/health"}
