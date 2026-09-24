"""API tests.

The FastAPI app is exercised through ``TestClient`` with the chain and stores
stubbed, so the tests cover routing, validation and serialisation without a
running PostgreSQL, Qdrant or Ollama.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ragbuilder.api.models import IngestRequest, QueryRequest  # noqa: E402
from ragbuilder.rag.chain import RagAnswer, Source  # noqa: E402


def make_answer(**overrides) -> RagAnswer:
    defaults = dict(
        question="Wie viele Urlaubstage?",
        answer="Der Urlaub beträgt jährlich mindestens 24 Werktage [1].",
        sources=[
            Source(
                index=1,
                chunk_id="burlg_00001_abc",
                doc_id="burlg",
                title="Bundesurlaubsgesetz",
                section="§ 3",
                page_start=2,
                page_end=2,
                citation="Bundesurlaubsgesetz, § 3, p. 2",
                excerpt="Der Urlaub beträgt jährlich mindestens 24 Werktage.",
                score=0.0163,
                retriever="hybrid",
                dense_rank=1,
                sparse_rank=1,
                cited=True,
            )
        ],
        confidence=0.91,
        refused=False,
        strategy="sentence",
        retrieval_mode="hybrid",
        model="mistral:latest",
        language="German",
        expanded_queries=["urlaubstage"],
        retrieval_ms=42.0,
        generation_ms=1800.0,
        total_ms=1850.0,
    )
    defaults.update(overrides)
    return RagAnswer(**defaults)


@pytest.fixture
def client(monkeypatch):
    from ragbuilder.api import main as api_main

    class StubChain:
        def __init__(self):
            self.calls = []

        def ask(self, question, **kwargs):
            self.calls.append((question, kwargs))
            return make_answer(question=question)

    stub = StubChain()
    monkeypatch.setattr(api_main, "get_chain", lambda: stub)
    monkeypatch.setattr(
        api_main.db,
        "list_documents",
        lambda: [
            {
                "doc_id": "burlg",
                "title": "Bundesurlaubsgesetz",
                "category": "leave",
                "language": "de",
                "page_count": 8,
                "char_count": 14000,
                "embedded_chunks": 12,
                "total_chunks": 12,
                "ingested_at": "2026-07-27T09:00:00Z",
            },
        ],
    )
    monkeypatch.setattr(
        api_main.db,
        "chunk_stats",
        lambda: [
            {
                "strategy": "sentence",
                "chunks": 500,
                "embedded": 500,
                "documents": 24,
                "avg_tokens": 310,
                "min_tokens": 20,
                "max_tokens": 512,
            },
        ],
    )
    monkeypatch.setattr(api_main.db, "wait_for_postgres", lambda *a, **k: None)
    monkeypatch.setattr(api_main.db, "init_schema", lambda *a, **k: None)

    with TestClient(api_main.app) as test_client:
        test_client.stub_chain = stub
        yield test_client


class TestQueryEndpoint:
    def test_returns_an_answer_with_sources(self, client):
        response = client.post("/query", json={"question": "Wie viele Urlaubstage stehen mir zu?"})
        assert response.status_code == 200
        body = response.json()
        assert "24 Werktage" in body["answer"]
        assert body["sources"][0]["citation"] == "Bundesurlaubsgesetz, § 3, p. 2"
        assert body["cited_source_count"] == 1
        assert body["confidence"] == pytest.approx(0.91)

    def test_passes_options_through_to_the_chain(self, client):
        client.post(
            "/query",
            json={
                "question": "Eine Frage zum Urlaub",
                "strategy": "fixed",
                "mode": "dense",
                "top_k": 3,
                "use_memory": False,
            },
        )
        _, kwargs = client.stub_chain.calls[-1]
        assert kwargs["strategy"] == "fixed"
        assert kwargs["mode"] == "dense"
        assert kwargs["top_k"] == 3
        assert kwargs["use_memory"] is False

    @pytest.mark.parametrize(
        "payload",
        [
            {},  # missing question
            {"question": "ab"},  # below min_length
            {"question": "   "},  # blank after stripping
            {"question": "gültige Frage", "mode": "telepathy"},
            {"question": "gültige Frage", "strategy": "vibes"},
            {"question": "gültige Frage", "top_k": 0},
            {"question": "gültige Frage", "top_k": 99},
        ],
    )
    def test_rejects_invalid_requests(self, client, payload):
        assert client.post("/query", json=payload).status_code == 422

    def test_reports_a_refusal_honestly(self, client, monkeypatch):
        from ragbuilder.api import main as api_main

        class RefusingChain:
            def ask(self, question, **kwargs):
                return make_answer(
                    question=question,
                    answer="Ich weiß es nicht.",
                    refused=True,
                    confidence=0.05,
                    sources=[],
                )

        monkeypatch.setattr(api_main, "get_chain", lambda: RefusingChain())
        body = client.post("/query", json={"question": "Wie hoch ist die Körperschaftsteuer?"}).json()
        assert body["refused"] is True
        assert body["sources"] == []

    def test_returns_503_when_the_model_is_down(self, client, monkeypatch):
        from ragbuilder.api import main as api_main
        from ragbuilder.llm.client import LLMError

        class BrokenChain:
            def ask(self, question, **kwargs):
                raise LLMError("ollama unreachable")

        monkeypatch.setattr(api_main, "get_chain", lambda: BrokenChain())
        response = client.post("/query", json={"question": "Eine gültige Frage"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_adds_a_timing_header(self, client):
        response = client.post("/query", json={"question": "Eine gültige Frage"})
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


class TestDocumentsEndpoint:
    def test_lists_the_knowledge_base(self, client):
        body = client.get("/documents").json()
        assert body["total_documents"] == 1
        assert body["total_chunks"] == 12
        assert body["documents"][0]["doc_id"] == "burlg"


class TestStatsEndpoint:
    def test_reports_chunk_counts(self, client, monkeypatch):
        from ragbuilder.api import main as api_main

        monkeypatch.setattr(
            api_main,
            "get_store",
            lambda: type("S", (), {"info": lambda self: {"exists": True, "points": 500}})(),
        )
        body = client.get("/stats").json()
        assert body["chunks"][0]["strategy"] == "sentence"
        assert body["vector_store"]["points"] == 500


class TestIngestEndpoint:
    def test_accepts_and_schedules_the_run(self, client, monkeypatch):
        from ragbuilder.api import main as api_main

        monkeypatch.setattr(api_main, "_run_ingestion", lambda payload: None)
        response = client.post("/ingest", json={"strategies": ["sentence"], "download": False})
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_rejects_a_concurrent_run(self, client, monkeypatch):
        from ragbuilder.api import main as api_main

        monkeypatch.setitem(api_main._ingest_state, "running", True)
        assert client.post("/ingest", json={}).status_code == 409
        monkeypatch.setitem(api_main._ingest_state, "running", False)

    def test_rejects_an_unknown_strategy(self, client):
        assert client.post("/ingest", json={"strategies": ["telepathy"]}).status_code == 422


class TestOpenApi:
    def test_schema_documents_every_endpoint(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/query", "/query/stream", "/ingest", "/documents", "/health", "/stats"):
            assert path in paths

    def test_docs_are_served(self, client):
        assert client.get("/docs").status_code == 200


class TestModels:
    def test_question_is_stripped(self):
        assert QueryRequest(question="  Wie viele Urlaubstage?  ").question == "Wie viele Urlaubstage?"

    def test_defaults_are_sensible(self):
        request = QueryRequest(question="Eine Frage zum Urlaub")
        assert request.session_id == "default"
        assert request.use_memory is True
        assert request.strategy is None

    def test_ingest_defaults(self):
        request = IngestRequest()
        assert request.download is True
        assert request.embed is True
        assert request.force is False
