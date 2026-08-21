"""PostgreSQL access layer.

Thin, explicit SQL rather than an ORM: the queries here are the interesting
part of the system and hiding them behind a mapper would make the retrieval
behaviour harder to reason about, not easier.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

from ragbuilder.config import PROJECT_ROOT, get_config

LOG = logging.getLogger("ragbuilder.db")

SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


@contextmanager
def connection(dsn: str | None = None):
    conn = psycopg2.connect(dsn or get_config().postgres.dsn)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def cursor(dsn: str | None = None, dict_rows: bool = False) -> Iterator[Any]:
    with connection(dsn) as conn:
        with conn:
            factory = RealDictCursor if dict_rows else None
            with conn.cursor(cursor_factory=factory) as cur:
                yield cur


def init_schema(dsn: str | None = None) -> None:
    """Apply ``sql/schema.sql``. Idempotent - safe on every boot."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with cursor(dsn) as cur:
        cur.execute(sql)
    LOG.info("schema applied")


def wait_for_postgres(retries: int = 30, delay: float = 2.0, dsn: str | None = None) -> None:
    import time

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with connection(dsn):
                return
        except psycopg2.OperationalError as exc:
            last = exc
            LOG.warning("postgres not ready (%s/%s)", attempt, retries)
            time.sleep(delay)
    raise RuntimeError(f"postgres unreachable: {last}")


# ----------------------------------------------------------------- documents
UPSERT_DOCUMENT = """
INSERT INTO documents (doc_id, title, source_url, filename, language, category,
                       content_hash, page_count, char_count, table_count, metadata)
VALUES (%(doc_id)s, %(title)s, %(source_url)s, %(filename)s, %(language)s, %(category)s,
        %(content_hash)s, %(page_count)s, %(char_count)s, %(table_count)s, %(metadata)s)
ON CONFLICT (doc_id) DO UPDATE SET
    title        = EXCLUDED.title,
    source_url   = EXCLUDED.source_url,
    filename     = EXCLUDED.filename,
    language     = EXCLUDED.language,
    category     = EXCLUDED.category,
    content_hash = EXCLUDED.content_hash,
    page_count   = EXCLUDED.page_count,
    char_count   = EXCLUDED.char_count,
    table_count  = EXCLUDED.table_count,
    metadata     = EXCLUDED.metadata,
    updated_at   = now()
"""


def upsert_document(document: dict[str, Any], dsn: str | None = None) -> None:
    payload = dict(document)
    payload["metadata"] = Json(payload.get("metadata") or {})
    with cursor(dsn) as cur:
        cur.execute(UPSERT_DOCUMENT, payload)


def get_document_hash(doc_id: str, dsn: str | None = None) -> str | None:
    with cursor(dsn) as cur:
        cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
        return row[0] if row else None


def list_documents(dsn: str | None = None) -> list[dict[str, Any]]:
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute("SELECT * FROM v_knowledge_base ORDER BY title")
        return [dict(row) for row in cur.fetchall()]


def delete_document(doc_id: str, dsn: str | None = None) -> None:
    """Cascades to pages and chunks."""
    with cursor(dsn) as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))


# --------------------------------------------------------------------- pages
def replace_pages(doc_id: str, pages: Sequence[dict[str, Any]], dsn: str | None = None) -> int:
    rows = [
        (doc_id, p["page_number"], p["text"], p.get("section"), len(p["text"]), p.get("has_tables", False))
        for p in pages
    ]
    with cursor(dsn) as cur:
        cur.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        if rows:
            execute_values(
                cur,
                "INSERT INTO pages (doc_id, page_number, text, section, char_count, has_tables) VALUES %s",
                rows,
                page_size=200,
            )
    return len(rows)


def get_pages(doc_id: str, dsn: str | None = None) -> list[dict[str, Any]]:
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute(
            "SELECT page_number, text, section, has_tables FROM pages WHERE doc_id = %s ORDER BY page_number",
            (doc_id,),
        )
        return [dict(row) for row in cur.fetchall()]


# -------------------------------------------------------------------- chunks
def replace_chunks(
    doc_id: str, strategy: str, chunks: Sequence[dict[str, Any]], dsn: str | None = None
) -> int:
    rows = [
        (
            c["chunk_id"],
            doc_id,
            strategy,
            c["chunk_index"],
            c["text"],
            c.get("token_count", 0),
            len(c["text"]),
            c.get("page_start"),
            c.get("page_end"),
            c.get("section"),
        )
        for c in chunks
    ]
    with cursor(dsn) as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s AND strategy = %s", (doc_id, strategy))
        if rows:
            execute_values(
                cur,
                """
                INSERT INTO chunks (chunk_id, doc_id, strategy, chunk_index, text,
                                    token_count, char_count, page_start, page_end, section)
                VALUES %s
                """,
                rows,
                page_size=500,
            )
    return len(rows)


def iter_chunks(strategy: str, only_unembedded: bool = False, dsn: str | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.token_count,
               c.page_start, c.page_end, c.section, d.title, d.language, d.category
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.strategy = %s
    """
    if only_unembedded:
        sql += " AND c.embedded = FALSE"
    sql += " ORDER BY c.doc_id, c.chunk_index"
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute(sql, (strategy,))
        return [dict(row) for row in cur.fetchall()]


def get_chunks_by_id(chunk_ids: Sequence[str], dsn: str | None = None) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end, c.section,
                   d.title, d.source_url, d.category
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (list(chunk_ids),),
        )
        return {row["chunk_id"]: dict(row) for row in cur.fetchall()}


def mark_embedded(chunk_ids: Sequence[str], dsn: str | None = None) -> None:
    if not chunk_ids:
        return
    with cursor(dsn) as cur:
        cur.execute("UPDATE chunks SET embedded = TRUE WHERE chunk_id = ANY(%s)", (list(chunk_ids),))


def reset_embedded(strategy: str, dsn: str | None = None) -> None:
    with cursor(dsn) as cur:
        cur.execute("UPDATE chunks SET embedded = FALSE WHERE strategy = %s", (strategy,))


def chunk_stats(dsn: str | None = None) -> list[dict[str, Any]]:
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute(
            """
            SELECT strategy,
                   count(*)                       AS chunks,
                   count(*) FILTER (WHERE embedded) AS embedded,
                   round(avg(token_count))        AS avg_tokens,
                   min(token_count)               AS min_tokens,
                   max(token_count)               AS max_tokens,
                   count(DISTINCT doc_id)         AS documents
            FROM chunks GROUP BY strategy ORDER BY strategy
            """
        )
        return [dict(row) for row in cur.fetchall()]


# --------------------------------------------------------------- run logging
def start_run(corpus: str, strategy: str, dsn: str | None = None) -> int:
    with cursor(dsn) as cur:
        cur.execute(
            "INSERT INTO ingestion_runs (corpus, strategy) VALUES (%s, %s) RETURNING run_id",
            (corpus, strategy),
        )
        return int(cur.fetchone()[0])


def finish_run(run_id: int, stats: dict[str, Any], error: str | None = None, dsn: str | None = None) -> None:
    with cursor(dsn) as cur:
        cur.execute(
            """
            UPDATE ingestion_runs SET
                finished_at = now(),
                documents_seen = %s, documents_new = %s, documents_changed = %s,
                documents_skipped = %s, chunks_written = %s,
                status = %s, error = %s
            WHERE run_id = %s
            """,
            (
                stats.get("seen", 0),
                stats.get("new", 0),
                stats.get("changed", 0),
                stats.get("skipped", 0),
                stats.get("chunks", 0),
                "failed" if error else "success",
                error,
                run_id,
            ),
        )


def log_query(entry: dict[str, Any], dsn: str | None = None) -> None:
    with cursor(dsn) as cur:
        cur.execute(
            """
            INSERT INTO query_log (session_id, question, answer, strategy, retrieval_mode,
                                   confidence, refused, chunk_ids, latency_ms, llm_model)
            VALUES (%(session_id)s, %(question)s, %(answer)s, %(strategy)s, %(retrieval_mode)s,
                    %(confidence)s, %(refused)s, %(chunk_ids)s, %(latency_ms)s, %(llm_model)s)
            """,
            entry,
        )


def save_evaluation_rows(rows: Sequence[dict[str, Any]], dsn: str | None = None) -> int:
    if not rows:
        return 0
    values = [
        (
            r["run_label"],
            r["strategy"],
            r["retrieval_mode"],
            r["question_id"],
            r["question"],
            r.get("answer"),
            r.get("faithfulness"),
            r.get("answer_relevancy"),
            r.get("context_precision"),
            r.get("context_recall"),
            r.get("latency_ms"),
        )
        for r in rows
    ]
    with cursor(dsn) as cur:
        execute_values(
            cur,
            """
            INSERT INTO evaluation_results (run_label, strategy, retrieval_mode, question_id,
                                            question, answer, faithfulness, answer_relevancy,
                                            context_precision, context_recall, latency_ms)
            VALUES %s
            """,
            values,
        )
    return len(values)


def evaluation_summary(run_label: str | None = None, dsn: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM v_evaluation_summary"
    params: tuple = ()
    if run_label:
        sql += " WHERE run_label = %s"
        params = (run_label,)
    sql += " ORDER BY strategy, retrieval_mode"
    with cursor(dsn, dict_rows=True) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def path_relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
