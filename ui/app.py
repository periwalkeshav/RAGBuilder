"""RAGBuilder chat UI.

A thin client over the FastAPI backend - it holds no model, no database
connection and no retrieval logic, which is what lets it deploy to Streamlit
Community Cloud while the heavy backend stays wherever the documents are allowed
to live. That split is the deployment story for a GDPR-sensitive system: the
public part is a UI, the regulated part never leaves the host.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

st.set_page_config(
    page_title="RAGBuilder — German Labour Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    "Wie viele Urlaubstage stehen mir mindestens zu?",
    "Wie lange darf die werktägliche Arbeitszeit höchstens sein?",
    "Ab wann gilt der Kündigungsschutz?",
    "What is the statutory minimum annual leave in Germany?",
    "Welche Merkmale schützt das AGG vor Benachteiligung?",
]


# ------------------------------------------------------------------ backend
@st.cache_data(ttl=15, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def fetch_documents() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/documents", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_stats() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/stats", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def stream_answer(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with requests.post(
        f"{API_URL}/query/stream", json=payload, stream=True, timeout=REQUEST_TIMEOUT
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    continue


def trigger_ingest(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_URL}/ingest", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{int(time.time())}"
if "pending" not in st.session_state:
    st.session_state.pending = None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("⚖️ RAGBuilder")
    st.caption("Retrieval-augmented QA over German labour law — fully local models")

    health = fetch_health()
    if health is None:
        st.error(f"API unreachable at {API_URL}")
    else:
        badge = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}[health["status"]]
        st.markdown(f"**Status** {badge} `{health['status']}`")
        for name, component in health["components"].items():
            icon = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}[component["status"]]
            st.caption(f"{icon} **{name}** — {component['detail'] or component['status']}")

    st.divider()
    st.subheader("Retrieval")

    strategy = st.selectbox(
        "Chunking strategy",
        ["sentence", "fixed", "semantic"],
        index=0,
        help="Which pre-computed chunk set to search. All three are embedded in the same collection.",
    )
    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid", "dense", "sparse"],
        index=0,
        help="hybrid = dense vectors + BM25 keywords fused with Reciprocal Rank Fusion.",
    )
    top_k = st.slider("Sources to retrieve", 1, 10, 5)
    use_memory = st.toggle(
        "Conversation memory",
        value=True,
        help="Passes the last 3 turns so follow-up questions can use pronouns.",
    )

    st.divider()
    st.subheader("Knowledge base")

    documents = fetch_documents()
    if documents:
        st.metric("Documents", documents["total_documents"])
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", f"{documents['total_chunks']:,}")
        col_b.metric("Embedded", f"{documents['embedded_chunks']:,}")

        with st.expander(f"Browse {documents['total_documents']} document(s)"):
            for document in documents["documents"]:
                st.markdown(
                    f"**{document['doc_id'].upper()}** — {document['title']}  \n"
                    f"<span style='color:#888;font-size:0.8em'>"
                    f"{document['page_count']} pages · {document['total_chunks']} chunks · "
                    f"{document.get('category') or '—'}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No documents yet — ingest the corpus below.")

    with st.expander("Ingest documents"):
        st.caption(
            "Downloads the 24 German labour law statutes from gesetze-im-internet.de, "
            "parses, chunks and embeds them. Takes several minutes on first run."
        )
        ingest_strategies = st.multiselect(
            "Strategies to build", ["sentence", "fixed", "semantic"], default=["sentence"]
        )
        force = st.checkbox("Force re-ingest", value=False)
        if st.button("Start ingestion", use_container_width=True):
            try:
                result = trigger_ingest(
                    {"strategies": ingest_strategies, "download": True, "force": force, "embed": True}
                )
                st.success(result["detail"])
                st.cache_data.clear()
            except requests.HTTPError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text}")
            except requests.RequestException as exc:
                st.error(str(exc))

        stats = fetch_stats()
        if stats and stats.get("ingestion", {}).get("running"):
            st.warning("Ingestion in progress…")
        elif stats and stats.get("ingestion", {}).get("last_result"):
            st.caption(f"Last run: {stats['ingestion']['last_result']}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"ui-{int(time.time())}"
        st.rerun()

    st.caption(
        f"Model: `{(health or {}).get('config', {}).get('llm_model', '?')}`  \n"
        f"Embeddings: `{(health or {}).get('config', {}).get('embedding_model', '?')}`"
    )


# --------------------------------------------------------------------- main
st.title("German Labour Law Assistant")
st.caption(
    "Every answer is grounded in the statutes below and cites the paragraph and page it came from. "
    "When the sources do not cover a question, the assistant says so instead of guessing."
)

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    columns = st.columns(len(EXAMPLES[:3]))
    for column, example in zip(columns, EXAMPLES[:3]):
        if column.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()
    with st.expander("More examples"):
        for example in EXAMPLES[3:]:
            if st.button(example, key=f"ex-{example}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()


def render_sources(sources: list[dict[str, Any]], confidence: float) -> None:
    if not sources:
        return
    cited = [s for s in sources if s.get("cited")]
    label = f"📚 {len(sources)} source(s) · {len(cited)} cited · retrieval confidence {confidence:.0%}"
    with st.expander(label):
        for source in sources:
            marker = "✅ cited" if source.get("cited") else "· retrieved"
            ranks = []
            if source.get("dense_rank"):
                ranks.append(f"dense #{source['dense_rank']}")
            if source.get("sparse_rank"):
                ranks.append(f"BM25 #{source['sparse_rank']}")
            st.markdown(
                f"**[{source['index']}] {source['citation']}**  \n"
                f"<span style='color:#888;font-size:0.8em'>{marker} · "
                f"{' · '.join(ranks) or source.get('retriever', '')} · "
                f"score {source['score']:.4f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#1A1D24;padding:0.75rem;border-radius:6px;"
                f"border-left:3px solid #4F8BF9;font-size:0.85em;margin-bottom:0.75rem'>"
                f"{source['excerpt']}</div>",
                unsafe_allow_html=True,
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message.get("confidence", 0.0))
            if message.get("meta"):
                st.caption(message["meta"])


question = st.chat_input("Ask about German labour law…") or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_slot = st.container()
        meta_slot = st.empty()

        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "strategy": strategy,
            "mode": mode,
            "top_k": top_k,
            "use_memory": use_memory,
        }

        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        confidence = 0.0
        meta = ""

        try:
            with st.spinner("Retrieving…"):
                events = stream_answer(payload)
                first = next(events, None)
                if first and first["type"] == "sources":
                    sources = first["sources"]
                    confidence = first["confidence"]
                    with sources_slot:
                        render_sources(sources, confidence)

            for event in events:
                if event["type"] == "token":
                    answer_parts.append(event["token"])
                    placeholder.markdown("".join(answer_parts) + "▌")
                elif event["type"] == "error":
                    st.error(event["error"])
                elif event["type"] == "done":
                    meta = (
                        f"{mode}/{strategy} · confidence {event.get('confidence', 0):.0%} · "
                        f"{event.get('model', '')} · {event.get('total_ms', 0) / 1000:.1f}s"
                    )
                    if event.get("refused"):
                        meta = "⚠️ refused — insufficient evidence in the sources · " + meta

            placeholder.markdown("".join(answer_parts))
            meta_slot.caption(meta)

        except requests.RequestException as exc:
            placeholder.error(f"Request failed: {exc}")
            answer_parts = [f"Request failed: {exc}"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "".join(answer_parts),
            "sources": sources,
            "confidence": confidence,
            "meta": meta,
        }
    )
