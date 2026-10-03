"""End-to-end integration tests.

Skipped unless the stack is up and the corpus has been ingested::

    docker compose up -d
    make ingest
    RUN_INTEGRATION=1 pytest tests/test_integration.py -v

These are the tests that would catch a real regression in the pipeline: that a
statutory question retrieves the right statute, that BM25 finds a literal
paragraph reference dense search would miss, and that the system refuses a
question its corpus cannot answer.
"""

from __future__ import annotations

import os

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 with the stack running and the corpus ingested",
)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("INTEGRATION_TIMEOUT", "300"))


def ask(question: str, **options) -> dict:
    response = requests.post(
        f"{API_URL}/query", json={"question": question, "use_memory": False, **options}, timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


class TestHealth:
    def test_all_components_are_reachable(self):
        body = requests.get(f"{API_URL}/health", timeout=30).json()
        assert body["status"] in {"ok", "degraded"}
        for name in ("postgres", "qdrant", "ollama"):
            assert body["components"][name]["status"] != "down", body["components"][name]

    def test_the_knowledge_base_is_populated(self):
        body = requests.get(f"{API_URL}/documents", timeout=30).json()
        assert body["total_documents"] >= 20
        assert body["embedded_chunks"] > 0


class TestRetrievalQuality:
    def test_a_leave_question_retrieves_the_leave_statute(self):
        body = ask("Wie viele Urlaubstage stehen mir mindestens zu?")
        assert "burlg" in [s["doc_id"] for s in body["sources"]]

    def test_a_working_time_question_retrieves_the_working_time_statute(self):
        body = ask("Wie lange darf die werktägliche Arbeitszeit höchstens sein?")
        assert "arbzg" in [s["doc_id"] for s in body["sources"]]

    def test_an_english_question_retrieves_the_german_statute(self):
        """The multilingual embedding model earning its keep."""
        body = ask("What is the statutory minimum annual leave in Germany?")
        assert "burlg" in [s["doc_id"] for s in body["sources"]]

    def test_hybrid_finds_a_literal_paragraph_reference(self):
        """BM25's contribution: an exact token dense search tends to blur."""
        body = ask("§ 3 Mindesturlaub", mode="hybrid")
        assert body["sources"]
        assert any(s.get("sparse_rank") for s in body["sources"])

    def test_every_source_carries_a_page_number(self):
        body = ask("Wie viele Urlaubstage stehen mir mindestens zu?")
        for source in body["sources"]:
            assert source["page_start"] is not None
            assert source["citation"]


class TestAnswerQuality:
    def test_the_answer_is_grounded_and_cited(self):
        body = ask("Wie viele Urlaubstage stehen mir mindestens zu?")
        assert not body["refused"]
        assert body["cited_source_count"] >= 1
        assert "24" in body["answer"]

    def test_an_out_of_corpus_question_is_refused(self):
        """The behaviour that separates a usable system from a plausible one."""
        body = ask("Wie hoch ist der Körperschaftsteuersatz für Kapitalgesellschaften?")
        answer = body["answer"].lower()
        assert body["refused"] or "weiß es nicht" in answer or "keine information" in answer

    def test_confidence_is_higher_for_a_covered_question(self):
        covered = ask("Wie viele Urlaubstage stehen mir mindestens zu?")
        uncovered = ask("What are the Basel III capital requirements for banks?")
        assert covered["confidence"] > uncovered["confidence"]


class TestRetrievalModes:
    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
    def test_every_mode_returns_results(self, mode):
        body = ask("Wie viele Urlaubstage stehen mir zu?", mode=mode)
        assert body["sources"]
        assert body["retrieval_mode"] == mode

    @pytest.mark.parametrize("strategy", ["fixed", "sentence", "semantic"])
    def test_every_ingested_strategy_is_searchable(self, strategy):
        body = requests.post(
            f"{API_URL}/query",
            json={"question": "Wie viele Urlaubstage stehen mir zu?", "strategy": strategy},
            timeout=TIMEOUT,
        )
        if body.status_code == 503:
            pytest.skip(f"strategy {strategy!r} has not been ingested")
        assert body.json()["strategy"] == strategy


class TestStreaming:
    def test_sources_arrive_before_the_first_token(self):
        import json

        with requests.post(
            f"{API_URL}/query/stream",
            json={"question": "Wie viele Urlaubstage stehen mir zu?", "use_memory": False},
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            response.raise_for_status()
            types = []
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    types.append(json.loads(line[6:])["type"])
                if len(types) >= 3:
                    break
        assert types[0] == "sources"
        assert "token" in types


class TestConversationMemory:
    def test_a_follow_up_resolves_against_the_previous_turn(self):
        session = "integration-memory"
        requests.post(
            f"{API_URL}/query",
            json={
                "question": "Wie viele Urlaubstage stehen mir mindestens zu?",
                "session_id": session,
                "use_memory": True,
            },
            timeout=TIMEOUT,
        ).raise_for_status()
        body = requests.post(
            f"{API_URL}/query",
            json={
                "question": "Und wann muss dieser genommen werden?",
                "session_id": session,
                "use_memory": True,
            },
            timeout=TIMEOUT,
        ).json()
        assert body["sources"]
