"""Retrieval tests: RRF arithmetic, BM25 tokenisation, query expansion, confidence."""

from __future__ import annotations

import pytest

from ragbuilder.retrieval.fusion import (
    RetrievedChunk,
    merge_results,
    reciprocal_rank_fusion,
)
from ragbuilder.retrieval.sparse import tokenize


def chunk(chunk_id: str, score: float = 0.0, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, score=score, text=text, doc_id="d", title="T")


class TestReciprocalRankFusion:
    def test_score_matches_the_formula(self):
        [(item, score)] = reciprocal_rank_fusion([["a"]], k=60)
        assert item == "a"
        assert score == pytest.approx(1 / 61)

    def test_agreement_beats_a_single_first_place(self):
        """The property that makes RRF worth using.

        ``b`` is 2nd and 2nd; ``a`` is 1st in one list and absent from the other.
        Agreement across independent retrievers should win.
        """
        fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60))
        assert fused["b"] > fused["a"]
        assert fused["b"] == pytest.approx(2 / 62)

    def test_ranking_is_by_position_not_score(self):
        """Raw scores never enter the calculation - only ranks do."""
        ranked = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
        assert [item for item, _ in ranked] == ["x", "y", "z"]

    def test_k_damps_the_influence_of_the_top_rank(self):
        small_k = dict(reciprocal_rank_fusion([["a"], ["b", "a"]], k=1))
        large_k = dict(reciprocal_rank_fusion([["a"], ["b", "a"]], k=1000))
        # With a small k the first place dominates; with a large k the gap closes.
        assert (small_k["a"] - small_k["b"]) > (large_k["a"] - large_k["b"])

    def test_weights_shift_the_balance(self):
        balanced = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
        assert balanced["a"] == pytest.approx(balanced["b"])
        weighted = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
        assert weighted["a"] > weighted["b"]

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_output_is_deterministic_on_ties(self):
        first = reciprocal_rank_fusion([["a", "b"]], k=60)
        second = reciprocal_rank_fusion([["a", "b"]], k=60)
        assert first == second

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []


class TestMergeResults:
    def test_labels_where_each_result_came_from(self):
        dense = [chunk("a", 0.9), chunk("b", 0.8)]
        sparse = [chunk("b", 12.0), chunk("c", 9.0)]
        merged = merge_results(dense, sparse, top_k=3)

        by_id = {c.chunk_id: c for c in merged}
        assert by_id["b"].retriever == "hybrid"
        assert by_id["a"].retriever == "dense"
        assert by_id["c"].retriever == "sparse"

    def test_keeps_both_original_scores(self):
        merged = merge_results([chunk("b", 0.83)], [chunk("b", 14.2)], top_k=1)
        assert merged[0].dense_score == pytest.approx(0.83)
        assert merged[0].sparse_score == pytest.approx(14.2)
        # The fused score is on the RRF scale, not either original scale.
        assert merged[0].score < 1.0

    def test_ranks_are_recorded(self):
        merged = merge_results([chunk("a"), chunk("b")], [chunk("b"), chunk("a")], top_k=2)
        by_id = {c.chunk_id: c for c in merged}
        assert by_id["a"].dense_rank == 1 and by_id["a"].sparse_rank == 2
        assert by_id["b"].dense_rank == 2 and by_id["b"].sparse_rank == 1

    def test_top_k_is_honoured(self):
        dense = [chunk(f"d{i}") for i in range(10)]
        sparse = [chunk(f"s{i}") for i in range(10)]
        assert len(merge_results(dense, sparse, top_k=4)) == 4

    def test_one_empty_retriever_still_works(self):
        merged = merge_results([chunk("a"), chunk("b")], [], top_k=2)
        assert [c.chunk_id for c in merged] == ["a", "b"]
        assert all(c.retriever == "dense" for c in merged)

    def test_both_empty(self):
        assert merge_results([], [], top_k=5) == []


class TestCitationFormatting:
    def test_single_page(self):
        c = RetrievedChunk(chunk_id="x", title="BUrlG", section="§ 3", page_start=2, page_end=2)
        assert c.citation == "BUrlG, § 3, p. 2"

    def test_page_range(self):
        c = RetrievedChunk(chunk_id="x", title="BUrlG", section="§ 3", page_start=2, page_end=4)
        assert c.citation == "BUrlG, § 3, pp. 2-4"

    def test_without_a_section(self):
        c = RetrievedChunk(chunk_id="x", title="BUrlG", page_start=1, page_end=1)
        assert c.citation == "BUrlG, p. 1"

    def test_falls_back_to_doc_id(self):
        c = RetrievedChunk(chunk_id="x", doc_id="burlg", page_start=1, page_end=1)
        assert c.citation.startswith("burlg")


class TestBM25Tokenisation:
    def test_keeps_the_section_symbol(self):
        assert "§" in tokenize("§ 622 Absatz 2")

    def test_keeps_paragraph_numbers(self):
        tokens = tokenize("§ 622 Absatz 2")
        assert "622" in tokens
        # This is the whole reason for hybrid search: an embedding model has no
        # reliable way to distinguish § 622 from § 623.
        assert "622" in tokens and "623" not in tokens

    def test_drops_german_stopwords(self):
        tokens = tokenize("Der Urlaub und die Arbeitszeit")
        assert "der" not in tokens and "und" not in tokens
        assert "urlaub" in tokens and "arbeitszeit" in tokens

    def test_drops_english_stopwords(self):
        tokens = tokenize("What is the minimum annual leave")
        assert "the" not in tokens and "is" not in tokens
        assert "minimum" in tokens and "leave" in tokens

    def test_lowercases(self):
        assert tokenize("URLAUB Werktage") == ["urlaub", "werktage"]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("der die das") == []


class TestQueryExpansion:
    @staticmethod
    def _retriever():
        """A Retriever with its heavy dependencies bypassed.

        ``expand_query`` is pure string work, so constructing the real class
        without touching the encoder or vector store keeps the test at unit speed.
        """
        from ragbuilder.retrieval.retriever import Retriever

        instance = Retriever.__new__(Retriever)
        from ragbuilder.config import Config

        instance.config = Config()
        return instance

    def test_normalises_paragraph_words_to_symbols(self):
        expansions = self._retriever().expand_query("Was steht in Paragraf 622 Absatz 2?")
        assert any("§" in e for e in expansions)

    def test_strips_the_question_frame(self):
        expansions = self._retriever().expand_query("Wie lange ist die Kündigungsfrist?")
        assert any(
            e.lower().startswith("die kündigungsfrist") or e.lower().startswith("kündigungsfrist")
            for e in expansions
        )

    def test_produces_a_keyword_only_variant(self):
        expansions = self._retriever().expand_query("Wie viele Urlaubstage stehen mir zu?")
        assert any("urlaubstage" in e.lower() and "wie" not in e.lower() for e in expansions)

    def test_respects_the_variant_cap(self):
        assert len(self._retriever().expand_query("Wie lange ist die Kündigungsfrist?", variants=1)) <= 1

    def test_never_repeats_the_original(self):
        query = "Kündigungsfrist"
        assert query.lower() not in [e.lower() for e in self._retriever().expand_query(query)]

    def test_english_questions_are_handled(self):
        expansions = self._retriever().expand_query("What is the minimum annual leave?")
        assert expansions


class TestConfidence:
    @staticmethod
    def _result(chunks, mode="hybrid"):
        from ragbuilder.retrieval.retriever import RetrievalResult

        return RetrievalResult(query="q", chunks=chunks, mode=mode)

    def test_no_results_means_no_confidence(self):
        assert self._result([]).confidence == 0.0

    def test_a_strong_agreed_match_scores_high(self):
        c = chunk("a")
        c.dense_score, c.dense_rank, c.sparse_rank = 0.94, 1, 1
        assert self._result([c]).confidence > 0.85

    def test_a_weak_match_scores_low(self):
        c = chunk("a")
        c.dense_score, c.dense_rank = 0.68, 1
        assert self._result([c]).confidence < 0.2

    def test_agreement_raises_confidence(self):
        agreed = chunk("a")
        agreed.dense_score, agreed.dense_rank, agreed.sparse_rank = 0.85, 1, 1
        alone = chunk("b")
        alone.dense_score, alone.dense_rank = 0.85, 1
        assert self._result([agreed]).confidence > self._result([alone]).confidence

    def test_dense_only_mode_ignores_agreement(self):
        c = chunk("a")
        c.dense_score, c.dense_rank = 0.90, 1
        assert self._result([c], mode="dense").confidence == pytest.approx(0.80, abs=0.01)

    def test_sparse_only_mode_uses_bm25_saturation(self):
        """Sparse mode has no dense score at all.

        Reading only dense_score made confidence 0 for every sparse query, so
        the mode refused everything - caught by the retrieval evaluation grid,
        not by a unit test, which is why this one exists now.
        """
        c = chunk("a")
        c.sparse_score, c.sparse_rank = 24.0, 1
        confidence = self._result([c], mode="sparse").confidence
        assert confidence > 0.5
        assert confidence < 1.0

    def test_a_weak_bm25_match_scores_low(self):
        c = chunk("a")
        c.sparse_score, c.sparse_rank = 1.5, 1
        assert self._result([c], mode="sparse").confidence < 0.2

    def test_confidence_stays_within_bounds(self):
        c = chunk("a")
        c.dense_score, c.dense_rank, c.sparse_rank = 1.5, 1, 1  # impossible, but must not overflow
        assert 0.0 <= self._result([c]).confidence <= 1.0
