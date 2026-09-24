"""Evaluation tests: metric arithmetic and test set integrity.

The judge is stubbed. What is being tested is the metric *definitions* - that
faithfulness is the supported fraction of claims, that an undecidable sample is
excluded rather than scored zero, and that the aggregate ignores ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragbuilder.evaluation.metrics import (
    MetricScores,
    aggregate,
    context_precision,
    faithfulness,
)
from ragbuilder.evaluation.runner import SampleResult, RunResult


class FakeJudge:
    """A judge with scripted verdicts."""

    def __init__(self, claims=None, supported=None, relevant=None, question=None):
        self._claims = claims if claims is not None else ["claim a", "claim b"]
        self._supported = list(supported) if supported is not None else [True, True]
        self._relevant = list(relevant) if relevant is not None else [True, True]
        self._question = question
        self.parse_failures = 0

    def extract_claims(self, answer):
        return list(self._claims)

    def claim_supported(self, claim, context):
        return self._supported.pop(0) if self._supported else None

    def passage_relevant(self, question, passage):
        return self._relevant.pop(0) if self._relevant else None

    def generated_question(self, answer):
        return self._question


class TestFaithfulness:
    def test_all_claims_supported_scores_one(self):
        score, detail = faithfulness("answer", ["ctx"], FakeJudge(supported=[True, True]))
        assert score == 1.0
        assert detail["supported"] == 2

    def test_no_claims_supported_scores_zero(self):
        score, _ = faithfulness("answer", ["ctx"], FakeJudge(supported=[False, False]))
        assert score == 0.0

    def test_half_supported_scores_half(self):
        score, detail = faithfulness("answer", ["ctx"], FakeJudge(supported=[True, False]))
        assert score == 0.5
        assert detail["unsupported_claims"] == ["claim b"]

    def test_undecidable_claims_are_excluded_not_counted_as_wrong(self):
        """A flaky judge must not look like a hallucinating system."""
        score, detail = faithfulness("answer", ["ctx"], FakeJudge(supported=[True, None]))
        assert score == 1.0
        assert detail["undecided"] == 1

    def test_no_claims_yields_no_score(self):
        score, detail = faithfulness("answer", ["ctx"], FakeJudge(claims=[]))
        assert score is None
        assert detail["claims"] == 0

    def test_all_verdicts_undecidable_yields_no_score(self):
        score, _ = faithfulness("answer", ["ctx"], FakeJudge(supported=[None, None]))
        assert score is None


class TestContextPrecision:
    def test_all_relevant_scores_one(self):
        score, _ = context_precision("q", ["a", "b"], FakeJudge(relevant=[True, True]))
        assert score == 1.0

    def test_half_relevant_scores_half(self):
        score, detail = context_precision("q", ["a", "b"], FakeJudge(relevant=[True, False]))
        assert score == 0.5
        assert detail["relevant"] == 1

    def test_no_contexts_yields_no_score(self):
        score, detail = context_precision("q", [], FakeJudge())
        assert score is None
        assert detail["retrieved"] == 0


class TestAggregate:
    def test_averages_available_values(self):
        scores = [MetricScores(faithfulness=1.0), MetricScores(faithfulness=0.0)]
        assert aggregate(scores)["faithfulness"] == 0.5

    def test_ignores_missing_values(self):
        scores = [MetricScores(faithfulness=1.0), MetricScores(faithfulness=None)]
        assert aggregate(scores)["faithfulness"] == 1.0

    def test_all_missing_yields_none(self):
        assert aggregate([MetricScores(), MetricScores()])["faithfulness"] is None

    def test_empty_input(self):
        assert aggregate([])["faithfulness"] is None


class TestSampleResult:
    def test_hit_when_an_expected_document_is_retrieved(self):
        sample = SampleResult("q1", "q", "a", expected_docs=["burlg"], retrieved_docs=["arbzg", "burlg"])
        assert sample.hit is True

    def test_miss_when_no_expected_document_is_retrieved(self):
        sample = SampleResult("q1", "q", "a", expected_docs=["burlg"], retrieved_docs=["arbzg"])
        assert sample.hit is False

    def test_no_expectation_means_no_hit_verdict(self):
        assert SampleResult("q1", "q", "a").hit is None

    def test_refusing_an_unanswerable_question_is_correct(self):
        assert SampleResult("q1", "q", "a", answerable=False, refused=True).refusal_correct

    def test_answering_an_unanswerable_question_is_wrong(self):
        assert not SampleResult("q1", "q", "a", answerable=False, refused=False).refusal_correct

    def test_refusing_an_answerable_question_is_wrong(self):
        assert not SampleResult("q1", "q", "a", answerable=True, refused=True).refusal_correct


class TestRunSummary:
    def test_reports_refusal_accuracy_and_citation_rate(self):
        run = RunResult(strategy="sentence", retrieval_mode="hybrid", label="t")
        run.samples = [
            SampleResult(
                "q1",
                "q",
                "a",
                answerable=True,
                refused=False,
                cited=True,
                scores=MetricScores(faithfulness=1.0),
            ),
            SampleResult(
                "q2",
                "q",
                "a",
                answerable=True,
                refused=False,
                cited=False,
                scores=MetricScores(faithfulness=0.5),
            ),
            SampleResult("q3", "q", "", answerable=False, refused=True),
        ]
        summary = run.summary()
        assert summary["questions"] == 3
        assert summary["refusal_accuracy"] == 1.0
        assert summary["citation_rate"] == 0.5
        assert summary["faithfulness"] == 0.75

    def test_empty_run_does_not_divide_by_zero(self):
        summary = RunResult(strategy="s", retrieval_mode="hybrid", label="t").summary()
        assert summary["questions"] == 0
        assert summary["refusal_accuracy"] is None


class TestTestset:
    @staticmethod
    def _questions():
        path = Path(__file__).resolve().parents[1] / "ragbuilder" / "evaluation" / "testset.json"
        return json.loads(path.read_text(encoding="utf-8"))["questions"]

    def test_has_at_least_twenty_questions(self):
        assert len(self._questions()) >= 20

    def test_ids_are_unique(self):
        ids = [q["id"] for q in self._questions()]
        assert len(set(ids)) == len(ids)

    def test_answerable_questions_have_a_ground_truth_and_expected_documents(self):
        for question in self._questions():
            if question.get("answerable", True):
                assert question["ground_truth"].strip(), question["id"]
                assert question["expected_docs"], question["id"]

    def test_unanswerable_questions_expect_nothing(self):
        unanswerable = [q for q in self._questions() if not q.get("answerable", True)]
        assert unanswerable, "the refusal path needs questions that cannot be answered"
        for question in unanswerable:
            assert not question["expected_docs"]
            assert not question["ground_truth"]

    def test_covers_both_languages(self):
        languages = {q["language"] for q in self._questions()}
        assert {"de", "en"} <= languages

    def test_expected_documents_exist_in_the_corpus(self):
        from ragbuilder.ingestion.corpora import get_corpus

        known = {d.doc_id for d in get_corpus("german_labour_law")}
        for question in self._questions():
            for doc_id in question["expected_docs"]:
                assert doc_id in known, f"{question['id']} expects unknown document {doc_id!r}"
