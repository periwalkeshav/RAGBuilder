"""Evaluation harness: run the test set across chunking strategies, log to MLflow.

The experiment this is built to answer: **does chunking strategy change answer
quality, and by how much?** Every strategy is scored on the same questions with
the same retrieval mode and the same judge, so the comparison is like for like.

Two metrics that are not in RAGAs are tracked as well, because they matter for a
system that is allowed to say no:

* ``refusal_accuracy`` - refused the unanswerable questions *and* answered the
  answerable ones. A system that refuses everything scores perfectly on
  faithfulness and is useless.
* ``citation_rate`` - fraction of answers carrying at least one ``[n]`` marker.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ragbuilder import db
from ragbuilder.config import PROJECT_ROOT, Config, get_config
from ragbuilder.evaluation.metrics import Judge, MetricScores, aggregate, evaluate_sample
from ragbuilder.llm.prompts import extract_citations
from ragbuilder.rag.chain import RagChain

LOG = logging.getLogger("ragbuilder.eval.runner")


@dataclass
class SampleResult:
    question_id: str
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    answerable: bool = True
    refused: bool = False
    confidence: float = 0.0
    cited: bool = False
    expected_docs: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    scores: MetricScores = field(default_factory=MetricScores)

    @property
    def hit(self) -> bool | None:
        """Did retrieval surface at least one expected document?"""
        if not self.expected_docs:
            return None
        return any(doc in self.retrieved_docs for doc in self.expected_docs)

    @property
    def refusal_correct(self) -> bool:
        return self.refused if not self.answerable else not self.refused


@dataclass
class RunResult:
    strategy: str
    retrieval_mode: str
    label: str
    samples: list[SampleResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        metrics = aggregate([s.scores for s in self.samples])
        answerable = [s for s in self.samples if s.answerable]
        hits = [s.hit for s in self.samples if s.hit is not None]

        metrics.update(
            {
                "questions": len(self.samples),
                "retrieval_hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
                "refusal_accuracy": (
                    round(sum(1 for s in self.samples if s.refusal_correct) / len(self.samples), 4)
                    if self.samples
                    else None
                ),
                "citation_rate": (
                    round(sum(1 for s in answerable if s.cited) / len(answerable), 4) if answerable else None
                ),
                "avg_confidence": (
                    round(sum(s.confidence for s in self.samples) / len(self.samples), 4)
                    if self.samples
                    else None
                ),
                "avg_latency_ms": (
                    round(sum(s.latency_ms for s in self.samples) / len(self.samples))
                    if self.samples
                    else None
                ),
                "duration_seconds": round(self.duration_seconds, 1),
            }
        )
        return metrics


def load_testset(path: str | Path | None = None, config: Config | None = None) -> list[dict]:
    config = config or get_config()
    testset_path = Path(path or config.evaluation.testset)
    if not testset_path.is_absolute():
        testset_path = PROJECT_ROOT / testset_path
    payload = json.loads(testset_path.read_text(encoding="utf-8"))
    return payload["questions"]


def run_strategy(
    strategy: str,
    questions: Sequence[dict],
    chain: RagChain,
    judge: Judge,
    retrieval_mode: str | None = None,
    label: str = "",
    score: bool = True,
    retrieval_only: bool = False,
) -> RunResult:
    """Score ``questions`` for one (strategy, retrieval mode) pair.

    ``retrieval_only`` skips generation entirely. Generation is ~40 s per
    question on CPU and retrieval is ~1 s, so tuning the retriever against the
    full grid of strategies and modes is a two-minute loop instead of an hour -
    and ``retrieval_hit_rate`` is the metric that actually moves when you change
    chunk size, fusion weights or expansion.
    """
    config = chain.config
    retrieval_mode = retrieval_mode or config.retrieval.strategy
    result = RunResult(strategy=strategy, retrieval_mode=retrieval_mode, label=label)
    started = time.perf_counter()

    for index, item in enumerate(questions, start=1):
        LOG.info(
            "[%s/%s] %s | %s: %s",
            index,
            len(questions),
            strategy,
            item["id"],
            item["question"][:70],
        )

        if retrieval_only:
            retrieval = chain.retriever.retrieve(item["question"], strategy=strategy, mode=retrieval_mode)
            result.samples.append(
                SampleResult(
                    question_id=item["id"],
                    question=item["question"],
                    answer="",
                    contexts=[c.text[:600] for c in retrieval.chunks],
                    ground_truth=item.get("ground_truth", ""),
                    answerable=item.get("answerable", True),
                    # Without generation, "would it have refused" is exactly the
                    # confidence gate the chain applies before calling the LLM.
                    refused=retrieval.confidence < config.rag.min_confidence,
                    confidence=retrieval.confidence,
                    cited=False,
                    expected_docs=item.get("expected_docs", []),
                    retrieved_docs=[c.doc_id for c in retrieval.chunks],
                    latency_ms=retrieval.latency_ms,
                )
            )
            continue

        answer = chain.ask(
            item["question"],
            session_id=f"eval-{strategy}",
            strategy=strategy,
            mode=retrieval_mode,
            use_memory=False,  # every question must be independent
            log=False,
        )

        contexts = [s.excerpt for s in answer.sources]
        sample = SampleResult(
            question_id=item["id"],
            question=item["question"],
            answer=answer.answer,
            contexts=contexts,
            ground_truth=item.get("ground_truth", ""),
            answerable=item.get("answerable", True),
            refused=answer.refused,
            confidence=answer.confidence,
            cited=bool(extract_citations(answer.answer, len(answer.sources))),
            expected_docs=item.get("expected_docs", []),
            retrieved_docs=[s.doc_id for s in answer.sources],
            latency_ms=answer.total_ms,
        )

        # Scoring a refusal is meaningless - there are no claims to check and no
        # answer to reconstruct a question from. Those samples contribute to
        # refusal_accuracy instead.
        if score and not sample.refused and sample.answerable:
            sample.scores = evaluate_sample(
                question=sample.question,
                answer=sample.answer,
                contexts=contexts,
                ground_truth=sample.ground_truth,
                judge=judge,
                encoder=chain.retriever.encoder,
            )

        result.samples.append(sample)

    result.duration_seconds = time.perf_counter() - started
    return result


def persist(result: RunResult) -> None:
    rows = [
        {
            "run_label": result.label,
            "strategy": result.strategy,
            "retrieval_mode": result.retrieval_mode,
            "question_id": s.question_id,
            "question": s.question,
            "answer": s.answer,
            "faithfulness": s.scores.faithfulness,
            "answer_relevancy": s.scores.answer_relevancy,
            "context_precision": s.scores.context_precision,
            "context_recall": s.scores.context_recall,
            "latency_ms": int(s.latency_ms),
        }
        for s in result.samples
    ]
    db.save_evaluation_rows(rows)


def log_to_mlflow(result: RunResult, config: Config) -> None:
    """Record one strategy's run as an MLflow experiment run."""
    try:
        import mlflow
    except ImportError:
        LOG.warning("mlflow is not installed; skipping experiment tracking")
        return

    try:
        mlflow.set_tracking_uri(config.evaluation.mlflow_uri)
        mlflow.set_experiment(config.evaluation.mlflow_experiment)

        with mlflow.start_run(run_name=f"{result.label}-{result.strategy}"):
            mlflow.log_params(
                {
                    "chunking_strategy": result.strategy,
                    "retrieval_mode": result.retrieval_mode,
                    "embedding_model": config.embedding.model,
                    "llm_model": config.llm.model,
                    "judge_model": config.evaluation.judge_model,
                    "top_k": config.retrieval.top_k,
                    "candidate_k": config.retrieval.candidate_k,
                    "rrf_k": config.retrieval.rrf_k,
                    "query_expansion": config.retrieval.query_expansion,
                    "min_confidence": config.rag.min_confidence,
                    "chunk_tokens": getattr(
                        getattr(config.chunking, result.strategy, None), "max_tokens", None
                    )
                    or config.chunking.fixed.chunk_tokens,
                }
            )
            metrics = {k: v for k, v in result.summary().items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {"samples": [_sample_payload(s) for s in result.samples]},
                f"samples_{result.strategy}.json",
            )
        LOG.info("logged %s to MLflow at %s", result.strategy, config.evaluation.mlflow_uri)
    except Exception as exc:  # noqa: BLE001 - tracking must not fail the evaluation
        LOG.warning("MLflow logging failed: %s", exc)


def _sample_payload(sample: SampleResult) -> dict:
    payload = asdict(sample)
    payload["scores"] = sample.scores.as_dict()
    payload["detail"] = sample.scores.detail
    payload["hit"] = sample.hit
    payload["refusal_correct"] = sample.refusal_correct
    # Contexts are large and already in PostgreSQL.
    payload["contexts"] = [c[:200] for c in payload["contexts"]]
    return payload


def run_evaluation(
    strategies: Sequence[str] | None = None,
    retrieval_modes: Sequence[str] | None = None,
    config: Config | None = None,
    label: str | None = None,
    limit: int | None = None,
    score: bool = True,
    mlflow_enabled: bool = True,
    retrieval_only: bool = False,
) -> dict[str, RunResult]:
    config = config or get_config()
    strategies = list(strategies or config.evaluation.strategies)
    retrieval_modes = list(retrieval_modes or [config.retrieval.strategy])
    label = label or time.strftime("%Y%m%d-%H%M%S")

    questions = load_testset(config=config)
    if limit:
        questions = questions[:limit]

    chain = RagChain(config)
    judge = Judge(config)
    results: dict[str, RunResult] = {}

    for strategy in strategies:
        for mode in retrieval_modes:
            key = f"{strategy}/{mode}"
            LOG.info("=" * 70)
            LOG.info("evaluating %s over %d question(s)", key, len(questions))
            LOG.info("=" * 70)
            result = run_strategy(
                strategy,
                questions,
                chain,
                judge,
                retrieval_mode=mode,
                label=label,
                score=score and not retrieval_only,
                retrieval_only=retrieval_only,
            )
            results[key] = result
            persist(result)
            if mlflow_enabled:
                log_to_mlflow(result, config)

    if judge.parse_failures:
        LOG.warning(
            "judge returned unparseable output %d time(s); those samples were excluded "
            "from the affected metric rather than scored as zero",
            judge.parse_failures,
        )
    return results


def format_table(results: dict[str, RunResult], retrieval_only: bool = False) -> str:
    """Markdown comparison table, ready to paste into the README."""
    if retrieval_only:
        header = (
            "| Strategy | Retrieval | Retrieval hit rate | Refusal accuracy | "
            "Avg confidence | Avg latency |"
        )
        divider = "|" + "---|" * 6
    else:
        header = (
            "| Strategy | Retrieval | Faithfulness | Answer relevancy | Context precision | "
            "Context recall | Retrieval hit rate | Refusal accuracy | Citation rate | Avg latency |"
        )
        divider = "|" + "---|" * 10

    def cell(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
        return f"{value:,}{suffix}"

    lines = [header, divider]
    for key, result in results.items():
        summary = result.summary()
        strategy, mode = key.split("/")
        if retrieval_only:
            lines.append(
                f"| {strategy} | {mode} | {cell(summary['retrieval_hit_rate'])} | "
                f"{cell(summary['refusal_accuracy'])} | {cell(summary['avg_confidence'])} | "
                f"{cell(summary['avg_latency_ms'])} ms |"
            )
            continue
        lines.append(
            f"| {strategy} | {mode} | {cell(summary['faithfulness'])} | "
            f"{cell(summary['answer_relevancy'])} | {cell(summary['context_precision'])} | "
            f"{cell(summary['context_recall'])} | {cell(summary['retrieval_hit_rate'])} | "
            f"{cell(summary['refusal_accuracy'])} | {cell(summary['citation_rate'])} | "
            f"{cell(summary['avg_latency_ms'])} ms |"
        )
    return "\n".join(lines)
