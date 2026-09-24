#!/usr/bin/env python3
"""RAGBuilder command line.

    python -m ragbuilder.cli init                     apply the database schema
    python -m ragbuilder.cli download                 fetch the corpus PDFs
    python -m ragbuilder.cli ingest --all-strategies  parse, chunk and store
    python -m ragbuilder.cli embed --all-strategies   encode chunks into Qdrant
    python -m ragbuilder.cli ask "Wie viele Urlaubstage stehen mir zu?"
    python -m ragbuilder.cli search "Kündigungsfrist" --mode hybrid
    python -m ragbuilder.cli evaluate --strategies fixed sentence semantic
    python -m ragbuilder.cli stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from ragbuilder import db
from ragbuilder.chunking.strategies import STRATEGIES
from ragbuilder.config import get_config

LOG = logging.getLogger("ragbuilder.cli")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level="DEBUG" if verbose else "INFO",
        format="%(asctime)s %(levelname)-7s %(name)-24s | %(message)s",
    )
    for noisy in ("urllib3", "httpx", "sentence_transformers", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------ commands
def cmd_init(args) -> int:
    db.wait_for_postgres()
    db.init_schema()
    print("schema applied")
    return 0


def cmd_download(args) -> int:
    from ragbuilder.ingestion.corpora import download_corpus

    config = get_config()
    corpus = args.corpus or config.corpus.name
    results = download_corpus(corpus, config.pdf_path, force=args.force)
    print(f"\n{len(results)} document(s) available in {config.pdf_path}")
    return 0


def cmd_ingest(args) -> int:
    from ragbuilder.embeddings.encoder import get_encoder
    from ragbuilder.ingestion.pipeline import ingest_corpus

    config = get_config()
    strategies = list(STRATEGIES) if args.all_strategies else (args.strategies or [config.chunking.strategy])

    # Semantic chunking needs the embedding model during chunking, not only
    # afterwards - it decides split points from sentence similarity.
    embed_fn = None
    if "semantic" in strategies:
        encoder = get_encoder(config)
        embed_fn = encoder.encode_plain

    db.wait_for_postgres()
    db.init_schema()
    stats = ingest_corpus(
        corpus_name=args.corpus,
        strategies=strategies,
        download=not args.no_download,
        force=args.force,
        embed_fn=embed_fn,
        limit=args.limit,
    )

    print("\ningestion summary")
    print("-----------------")
    for key, value in stats.as_dict().items():
        print(f"  {key:<18} {value}")
    if stats.failures:
        print("\n  failures:")
        for name, error in stats.failures:
            print(f"    {name}: {error}")
    return 0


def cmd_embed(args) -> int:
    from ragbuilder.embeddings.pipeline import embed_all, embed_strategy

    config = get_config()
    if args.all_strategies:
        results = embed_all(strategies=list(STRATEGIES), force=args.force)
    else:
        results = [
            embed_strategy(
                args.strategy or config.chunking.strategy,
                force=args.force,
                recreate_collection=args.recreate,
                prune=not args.no_prune,
            )
        ]

    print("\nembedding summary")
    print("-----------------")
    for result in results:
        print(
            f"  {result.strategy:<10} {result.chunks:>6,} chunk(s) "
            f"in {result.duration_seconds:>6.1f}s ({result.chunks_per_second:.1f}/s)"
        )
    return 0


def cmd_ask(args) -> int:
    from ragbuilder.rag.chain import get_chain

    chain = get_chain()
    started = time.perf_counter()
    answer = chain.ask(
        args.question,
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
        use_memory=False,
    )

    print("\n" + "=" * 78)
    print(f"Q: {answer.question}")
    print("=" * 78)
    print(answer.answer)
    print("-" * 78)
    print(
        f"confidence {answer.confidence:.2f} | {answer.retrieval_mode}/{answer.strategy} | "
        f"{answer.model} | retrieval {answer.retrieval_ms:.0f} ms | "
        f"generation {answer.generation_ms:.0f} ms | total {time.perf_counter() - started:.1f} s"
    )
    if answer.refused:
        print("REFUSED - retrieval confidence below threshold")
    print("\nsources:")
    for source in answer.sources:
        marker = "*" if source.cited else " "
        print(f" {marker}[{source.index}] {source.citation}  ({source.retriever}, score {source.score:.4f})")
        print(f"      {source.excerpt[:160]}...")
    if answer.expanded_queries:
        print(f"\nquery expansions: {answer.expanded_queries}")
    return 0


def cmd_search(args) -> int:
    from ragbuilder.retrieval.retriever import get_retriever

    result = get_retriever().retrieve(args.query, strategy=args.strategy, mode=args.mode, top_k=args.top_k)
    print(
        f"\n{len(result.chunks)} result(s) for {args.query!r} "
        f"[{result.mode}/{result.strategy}] in {result.latency_ms:.0f} ms "
        f"(confidence {result.confidence:.2f})"
    )
    if result.expanded_queries:
        print(f"expansions: {result.expanded_queries}")
    print("-" * 78)
    for position, chunk in enumerate(result.chunks, start=1):
        ranks = []
        if chunk.dense_rank:
            ranks.append(f"dense #{chunk.dense_rank}")
        if chunk.sparse_rank:
            ranks.append(f"sparse #{chunk.sparse_rank}")
        print(f"{position}. {chunk.citation}")
        print(f"   score {chunk.score:.4f}  [{', '.join(ranks) or chunk.retriever}]")
        print(f"   {chunk.text[:220].strip()}...")
        print()
    return 0


def cmd_evaluate(args) -> int:
    from ragbuilder.evaluation.runner import format_table, run_evaluation

    results = run_evaluation(
        strategies=args.strategies,
        retrieval_modes=args.modes,
        label=args.label,
        limit=args.limit,
        score=not args.no_score,
        mlflow_enabled=not args.no_mlflow,
        retrieval_only=args.retrieval_only,
    )

    print("\n" + format_table(results, retrieval_only=args.retrieval_only))
    if args.output:
        payload = {key: result.summary() for key, result in results.items()}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.output}")
    return 0


def cmd_stats(args) -> int:
    from ragbuilder.embeddings.store import get_store

    rows = db.chunk_stats()
    print("\nchunks by strategy")
    print("------------------")
    print(f"{'strategy':<12}{'chunks':>9}{'embedded':>10}{'docs':>7}{'avg tok':>9}{'min':>6}{'max':>6}")
    for row in rows:
        print(
            f"{row['strategy']:<12}{row['chunks']:>9,}{row['embedded']:>10,}"
            f"{row['documents']:>7}{int(row['avg_tokens'] or 0):>9}"
            f"{int(row['min_tokens'] or 0):>6}{int(row['max_tokens'] or 0):>6}"
        )

    print("\nvector store")
    print("------------")
    for key, value in get_store().info().items():
        print(f"  {key:<14} {value}")

    print("\ndocuments")
    print("---------")
    for document in db.list_documents():
        print(
            f"  {document['doc_id']:<14} {document['title'][:44]:<46}"
            f"{document['page_count']:>4} p  {document['total_chunks']:>5} chunks"
        )
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "ragbuilder.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbuilder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="apply the database schema").set_defaults(func=cmd_init)

    download = sub.add_parser("download", help="download the corpus PDFs")
    download.add_argument("--corpus")
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    ingest = sub.add_parser("ingest", help="parse, chunk and store documents")
    ingest.add_argument("--corpus")
    ingest.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    ingest.add_argument("--all-strategies", action="store_true")
    ingest.add_argument("--no-download", action="store_true")
    ingest.add_argument("--force", action="store_true")
    ingest.add_argument("--limit", type=int)
    ingest.set_defaults(func=cmd_ingest)

    embed = sub.add_parser("embed", help="encode chunks into Qdrant")
    embed.add_argument("--strategy", choices=STRATEGIES)
    embed.add_argument("--all-strategies", action="store_true")
    embed.add_argument("--force", action="store_true")
    embed.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    embed.add_argument("--no-prune", action="store_true", help="keep vectors whose chunk no longer exists")
    embed.set_defaults(func=cmd_embed)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=STRATEGIES)
    ask.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    ask.add_argument("--top-k", type=int)
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="retrieve without generating")
    search.add_argument("query")
    search.add_argument("--strategy", choices=STRATEGIES)
    search.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    evaluate = sub.add_parser("evaluate", help="score the test set across strategies")
    evaluate.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    evaluate.add_argument("--modes", nargs="+", choices=["dense", "sparse", "hybrid"])
    evaluate.add_argument("--label")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--output")
    evaluate.add_argument("--no-score", action="store_true", help="generate answers but skip LLM judging")
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip generation entirely - grades the retriever in seconds instead of an hour",
    )
    evaluate.add_argument("--no-mlflow", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    sub.add_parser("stats", help="knowledge base statistics").set_defaults(func=cmd_stats)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
