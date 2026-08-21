-- ============================================================================
--  RAGBuilder - PostgreSQL schema
--
--  Postgres is the system of record for documents, pages and chunks; Qdrant
--  holds only vectors plus a small payload. Keeping the text here means the
--  vector store can be dropped and rebuilt from scratch (different embedding
--  model, different chunking strategy) without re-parsing a single PDF.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ------------------------------------------------------------------ documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT        NOT NULL,
    source_url      TEXT,
    filename        TEXT        NOT NULL,
    language        TEXT        NOT NULL DEFAULT 'de',
    category        TEXT,
    -- SHA-256 of the file. Incremental ingestion compares this and skips a
    -- document whose bytes have not changed, which turns a re-run from a
    -- 10-minute job into a 2-second one.
    content_hash    TEXT        NOT NULL,
    page_count      INTEGER     NOT NULL DEFAULT 0,
    char_count      INTEGER     NOT NULL DEFAULT 0,
    table_count     INTEGER     NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_documents_language ON documents (language);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents (category);

-- ---------------------------------------------------------------------- pages
CREATE TABLE IF NOT EXISTS pages (
    doc_id      TEXT    NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    section     TEXT,
    char_count  INTEGER NOT NULL DEFAULT 0,
    has_tables  BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (doc_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_pages_section ON pages (section);
-- Trigram index so the sparse retriever can fall back to SQL when the BM25
-- index has not been built yet.
CREATE INDEX IF NOT EXISTS idx_pages_text_trgm ON pages USING gin (text gin_trgm_ops);

-- --------------------------------------------------------------------- chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT    NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    -- Which chunker produced this row. All three strategies coexist in the same
    -- table so the evaluation can compare them without re-ingesting.
    strategy     TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    token_count  INTEGER NOT NULL DEFAULT 0,
    char_count   INTEGER NOT NULL DEFAULT 0,
    page_start   INTEGER,
    page_end     INTEGER,
    section      TEXT,
    embedded     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, strategy, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks (strategy);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks (strategy, embedded);

-- ------------------------------------------------------------- ingestion log
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    corpus        TEXT,
    strategy      TEXT,
    documents_seen    INTEGER NOT NULL DEFAULT 0,
    documents_new     INTEGER NOT NULL DEFAULT 0,
    documents_changed INTEGER NOT NULL DEFAULT 0,
    documents_skipped INTEGER NOT NULL DEFAULT 0,
    chunks_written    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'running',
    error         TEXT
);

-- -------------------------------------------------------------- query log
-- Every answer is logged with its retrieval confidence and latency. This is
-- what makes the system observable rather than a black box.
CREATE TABLE IF NOT EXISTS query_log (
    query_id       BIGSERIAL PRIMARY KEY,
    session_id     TEXT,
    question       TEXT NOT NULL,
    answer         TEXT,
    strategy       TEXT,
    retrieval_mode TEXT,
    confidence     DOUBLE PRECISION,
    refused        BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_ids      TEXT[],
    latency_ms     INTEGER,
    llm_model      TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_log_session ON query_log (session_id);

-- ------------------------------------------------------- evaluation results
CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id           BIGSERIAL PRIMARY KEY,
    run_label         TEXT NOT NULL,
    strategy          TEXT NOT NULL,
    retrieval_mode    TEXT NOT NULL,
    question_id       TEXT NOT NULL,
    question          TEXT NOT NULL,
    answer            TEXT,
    faithfulness      DOUBLE PRECISION,
    answer_relevancy  DOUBLE PRECISION,
    context_precision DOUBLE PRECISION,
    context_recall    DOUBLE PRECISION,
    latency_ms        INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results (run_label, strategy);

-- ------------------------------------------------------------------- views
CREATE OR REPLACE VIEW v_knowledge_base AS
SELECT d.doc_id,
       d.title,
       d.category,
       d.language,
       d.page_count,
       d.char_count,
       d.ingested_at,
       count(c.chunk_id) FILTER (WHERE c.embedded)     AS embedded_chunks,
       count(c.chunk_id)                               AS total_chunks
FROM documents d
LEFT JOIN chunks c ON c.doc_id = d.doc_id
GROUP BY d.doc_id, d.title, d.category, d.language, d.page_count, d.char_count, d.ingested_at;

CREATE OR REPLACE VIEW v_evaluation_summary AS
SELECT run_label,
       strategy,
       retrieval_mode,
       count(*)                    AS questions,
       round(avg(faithfulness)::numeric, 4)      AS faithfulness,
       round(avg(answer_relevancy)::numeric, 4)  AS answer_relevancy,
       round(avg(context_precision)::numeric, 4) AS context_precision,
       round(avg(context_recall)::numeric, 4)    AS context_recall,
       round(avg(latency_ms)::numeric, 0)        AS avg_latency_ms
FROM evaluation_results
GROUP BY run_label, strategy, retrieval_mode;
