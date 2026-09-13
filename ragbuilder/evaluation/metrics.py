"""RAGAs metrics, implemented against a local judge model.

The four metrics follow the RAGAs definitions:

============================  =============================================================
Metric                        What it measures
============================  =============================================================
``faithfulness``              Of the claims in the answer, what fraction is supported by the
                              retrieved context. This is the hallucination rate, inverted.
``answer_relevancy``          Ask the judge to reconstruct the question from the answer,
                              then measure embedding similarity to the real question. A
                              rambling or evasive answer scores low even if it is faithful.
``context_precision``         Of the retrieved chunks, what fraction is actually useful.
                              This grades the *retriever*, not the generator.
``context_recall``            Of the ground-truth answer's sentences, what fraction the
                              retrieved context could support. Needs a reference answer.
============================  =============================================================

**Why not import the ``ragas`` package?** It defaults to OpenAI for judging and
for its embeddings. Sending this corpus and every generated answer to a
third-party API would contradict the entire premise of the project - local
models, nothing leaves the machine, defensible under GDPR. Implementing the
definitions against the local judge keeps that property, and it makes the metric
inspectable: when faithfulness drops you can read the per-claim verdicts instead
of trusting a number. ``--use-ragas`` still runs the real package if it is
installed and you have configured it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ragbuilder.config import Config, get_config
from ragbuilder.embeddings.encoder import get_encoder
from ragbuilder.llm.client import LLMError, get_client
from ragbuilder.llm.prompts import (
    CLAIM_EXTRACTION_TEMPLATE,
    CONTEXT_RECALL_TEMPLATE,
    CONTEXT_RELEVANCE_TEMPLATE,
    FAITHFULNESS_TEMPLATE,
    QUESTION_GENERATION_TEMPLATE,
)

LOG = logging.getLogger("ragbuilder.eval")

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class MetricScores:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


def _strip_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text).strip()


class Judge:
    """LLM-as-judge wrapper with defensive parsing.

    A 7B judge returns malformed JSON often enough that every call has to
    tolerate it. Unparseable verdicts are counted as ``None`` and excluded from
    the average rather than silently scored as zero, which would make a flaky
    judge look like a hallucinating system.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.client = get_client(self.config)
        self.model = self.config.evaluation.judge_model
        self.parse_failures = 0

    def _json(self, prompt: str) -> dict | list | None:
        try:
            return self.client.generate_json(prompt, model=self.model)
        except LLMError as exc:
            LOG.warning("judge call failed: %s", exc)
            self.parse_failures += 1
            return None

    def extract_claims(self, answer: str) -> list[str]:
        answer = _strip_citations(answer)
        if not answer:
            return []
        payload = self._json(CLAIM_EXTRACTION_TEMPLATE.render(answer=answer))
        if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
            claims = [str(c).strip() for c in payload["claims"] if str(c).strip()]
            if claims:
                return claims
        # Fall back to sentence splitting so a judge failure degrades the
        # granularity of the metric rather than dropping the sample.
        self.parse_failures += 1
        return [s.strip() for s in SENTENCE_SPLIT.split(answer) if len(s.strip()) > 15]

    def claim_supported(self, claim: str, context: str) -> bool | None:
        payload = self._json(FAITHFULNESS_TEMPLATE.render(claim=claim, context=context))
        if isinstance(payload, dict) and isinstance(payload.get("supported"), bool):
            return payload["supported"]
        self.parse_failures += 1
        return None

    def generated_question(self, answer: str) -> str | None:
        payload = self._json(QUESTION_GENERATION_TEMPLATE.render(answer=_strip_citations(answer)))
        if isinstance(payload, dict) and payload.get("question"):
            return str(payload["question"]).strip()
        self.parse_failures += 1
        return None

    def passage_relevant(self, question: str, passage: str) -> bool | None:
        payload = self._json(CONTEXT_RELEVANCE_TEMPLATE.render(question=question, passage=passage[:2500]))
        if isinstance(payload, dict) and isinstance(payload.get("relevant"), bool):
            return payload["relevant"]
        self.parse_failures += 1
        return None

    def recall_fraction(self, ground_truth: str, context: str) -> float | None:
        payload = self._json(
            CONTEXT_RECALL_TEMPLATE.render(ground_truth=ground_truth, context=context[:6000])
        )
        if isinstance(payload, dict):
            try:
                total = float(payload.get("total", 0))
                attributable = float(payload.get("attributable", 0))
                if total > 0:
                    return max(0.0, min(1.0, attributable / total))
            except (TypeError, ValueError):
                pass
        self.parse_failures += 1
        return None


# ---------------------------------------------------------------- the metrics
def faithfulness(answer: str, contexts: Sequence[str], judge: Judge) -> tuple[float | None, dict]:
    """Fraction of the answer's claims that the context supports."""
    claims = judge.extract_claims(answer)
    if not claims:
        return None, {"claims": 0}

    context = "\n\n".join(contexts)
    verdicts = [judge.claim_supported(claim, context) for claim in claims]
    decided = [v for v in verdicts if v is not None]
    if not decided:
        return None, {"claims": len(claims), "undecided": len(verdicts)}

    score = sum(1 for v in decided if v) / len(decided)
    return round(score, 4), {
        "claims": len(claims),
        "supported": sum(1 for v in decided if v),
        "undecided": len(verdicts) - len(decided),
        "unsupported_claims": [c for c, v in zip(claims, verdicts) if v is False],
    }


def answer_relevancy(question: str, answer: str, judge: Judge, encoder=None) -> tuple[float | None, dict]:
    """Similarity between the real question and one reconstructed from the answer."""
    reconstructed = judge.generated_question(answer)
    if not reconstructed:
        return None, {}

    encoder = encoder or get_encoder()
    vectors = encoder.encode_queries([question, reconstructed])
    similarity = float(
        np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]) + 1e-9)
    )
    # Cosine can be negative; relevancy is defined on 0-1.
    return round(max(0.0, similarity), 4), {"generated_question": reconstructed}


def context_precision(question: str, contexts: Sequence[str], judge: Judge) -> tuple[float | None, dict]:
    """Fraction of retrieved chunks that are actually useful for the question."""
    if not contexts:
        return None, {"retrieved": 0}
    verdicts = [judge.passage_relevant(question, c) for c in contexts]
    decided = [v for v in verdicts if v is not None]
    if not decided:
        return None, {"retrieved": len(contexts), "undecided": len(verdicts)}
    score = sum(1 for v in decided if v) / len(decided)
    return round(score, 4), {
        "retrieved": len(contexts),
        "relevant": sum(1 for v in decided if v),
        "relevance_mask": [bool(v) if v is not None else None for v in verdicts],
    }


def context_recall(ground_truth: str, contexts: Sequence[str], judge: Judge) -> tuple[float | None, dict]:
    """Fraction of the reference answer the retrieved context could support."""
    if not ground_truth or not contexts:
        return None, {}
    score = judge.recall_fraction(ground_truth, "\n\n".join(contexts))
    return (round(score, 4) if score is not None else None), {}


def evaluate_sample(
    question: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: str | None = None,
    judge: Judge | None = None,
    encoder=None,
) -> MetricScores:
    """Score one question/answer/context triple on all applicable metrics."""
    judge = judge or Judge()
    scores = MetricScores()

    scores.faithfulness, faith_detail = faithfulness(answer, contexts, judge)
    scores.answer_relevancy, relevancy_detail = answer_relevancy(question, answer, judge, encoder)
    scores.context_precision, precision_detail = context_precision(question, contexts, judge)
    if ground_truth:
        scores.context_recall, _ = context_recall(ground_truth, contexts, judge)

    scores.detail = {
        "faithfulness": faith_detail,
        "answer_relevancy": relevancy_detail,
        "context_precision": precision_detail,
    }
    return scores


def aggregate(scores: Sequence[MetricScores]) -> dict[str, float | None]:
    """Mean of each metric, ignoring samples the judge could not decide."""
    output: dict[str, float | None] = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        values = [getattr(s, name) for s in scores if getattr(s, name) is not None]
        output[name] = round(float(np.mean(values)), 4) if values else None
    return output
