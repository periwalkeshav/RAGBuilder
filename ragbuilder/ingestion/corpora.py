"""Document corpora and their download logic.

The default corpus is **German labour and employment law**, published by the
Federal Ministry of Justice at gesetze-im-internet.de. Three reasons it is a
good choice for this project and not just a convenient one:

* It is genuinely bilingual-hard. The text is German legalese; the questions a
  user asks may be German or English. That is exactly the case multilingual-e5
  is built for, and where an English-only embedding model visibly falls over.
* It is structured. Every law is divided into numbered paragraphs (``§ 1``,
  ``§ 2`` ...), which gives the chunkers a real section hierarchy to respect and
  makes citations checkable - you can look up "§ 622 BGB" and see whether the
  answer was right.
* It is unambiguously redistributable. Under § 5 UrhG, German laws and official
  decrees are free of copyright.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

LOG = logging.getLogger("ragbuilder.corpora")

USER_AGENT = "RAGBuilder/1.0 (portfolio project; contact via GitHub)"
BASE = "https://www.gesetze-im-internet.de"


@dataclass(frozen=True)
class Document:
    """One source document in a corpus."""

    doc_id: str
    title: str
    url: str
    filename: str
    language: str = "de"
    category: str = ""


def _law(slug: str, code: str, title: str, category: str) -> Document:
    return Document(
        doc_id=code.lower(),
        title=title,
        url=f"{BASE}/{slug}/{code}.pdf",
        filename=f"{code}.pdf",
        language="de",
        category=category,
    )


# 24 statutes covering the core of German employment law.
GERMAN_LABOUR_LAW: list[Document] = [
    _law("kschg", "KSchG", "Kündigungsschutzgesetz", "termination"),
    _law("arbzg", "ArbZG", "Arbeitszeitgesetz", "working time"),
    _law("burlg", "BUrlG", "Bundesurlaubsgesetz", "leave"),
    _law("tzbfg", "TzBfG", "Teilzeit- und Befristungsgesetz", "contracts"),
    _law("entgfg", "EntgFG", "Entgeltfortzahlungsgesetz", "pay"),
    _law("muschg_2018", "MuSchG", "Mutterschutzgesetz", "parental"),
    _law("beeg", "BEEG", "Bundeselterngeld- und Elternzeitgesetz", "parental"),
    _law("agg", "AGG", "Allgemeines Gleichbehandlungsgesetz", "equal treatment"),
    _law("betrvg", "BetrVG", "Betriebsverfassungsgesetz", "works councils"),
    _law("arbschg", "ArbSchG", "Arbeitsschutzgesetz", "health and safety"),
    _law("milog", "MiLoG", "Mindestlohngesetz", "pay"),
    _law("nachwg", "NachwG", "Nachweisgesetz", "contracts"),
    _law("bbig_2005", "BBiG", "Berufsbildungsgesetz", "training"),
    _law("jarbschg", "JArbSchG", "Jugendarbeitsschutzgesetz", "health and safety"),
    _law("pflegezg", "PflegeZG", "Pflegezeitgesetz", "leave"),
    _law("fpfzg", "FPfZG", "Familienpflegezeitgesetz", "leave"),
    _law("arbnerfg", "ArbnErfG", "Gesetz über Arbeitnehmererfindungen", "IP"),
    _law("a_g", "AÜG", "Arbeitnehmerüberlassungsgesetz", "agency work"),
    _law("tvg", "TVG", "Tarifvertragsgesetz", "collective agreements"),
    _law("arbgg", "ArbGG", "Arbeitsgerichtsgesetz", "procedure"),
    _law("betravg", "BetrAVG", "Betriebsrentengesetz", "pensions"),
    _law("bdsg_2018", "BDSG", "Bundesdatenschutzgesetz", "data protection"),
    _law("schwarzarbg_2004", "SchwarzArbG", "Schwarzarbeitsbekämpfungsgesetz", "compliance"),
    _law("bpersvg_2021", "BPersVG", "Bundespersonalvertretungsgesetz", "works councils"),
    _law("arbplschg", "ArbPlSchG", "Arbeitsplatzschutzgesetz", "job protection"),
    _law("geschgehg", "GeschGehG", "Geschäftsgeheimnisgesetz", "confidentiality"),
]

# A small English fallback so the pipeline can be demonstrated without German.
ARXIV_NLP: list[Document] = [
    Document(
        "arxiv_1706.03762",
        "Attention Is All You Need",
        "https://arxiv.org/pdf/1706.03762",
        "attention.pdf",
        "en",
        "nlp",
    ),
    Document(
        "arxiv_1810.04805",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "https://arxiv.org/pdf/1810.04805",
        "bert.pdf",
        "en",
        "nlp",
    ),
    Document(
        "arxiv_2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "https://arxiv.org/pdf/2005.11401",
        "rag.pdf",
        "en",
        "nlp",
    ),
    Document(
        "arxiv_2004.04906",
        "Dense Passage Retrieval for Open-Domain QA",
        "https://arxiv.org/pdf/2004.04906",
        "dpr.pdf",
        "en",
        "nlp",
    ),
    Document(
        "arxiv_2212.03533",
        "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
        "https://arxiv.org/pdf/2212.03533",
        "e5.pdf",
        "en",
        "nlp",
    ),
]

CORPORA: dict[str, list[Document]] = {
    "german_labour_law": GERMAN_LABOUR_LAW,
    "arxiv_nlp": ARXIV_NLP,
}


def get_corpus(name: str) -> list[Document]:
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; available: {', '.join(sorted(CORPORA))}")
    return CORPORA[name]


def download_corpus(
    name: str,
    destination: Path,
    force: bool = False,
    delay_seconds: float = 0.5,
    timeout: int = 60,
) -> list[tuple[Document, Path]]:
    """Download every document in ``name`` into ``destination``.

    Already-present files are skipped unless ``force``, so this is safe to run
    repeatedly. A small delay between requests keeps the load on a public
    government server negligible.
    """
    destination.mkdir(parents=True, exist_ok=True)
    documents = get_corpus(name)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    downloaded: list[tuple[Document, Path]] = []
    failures: list[tuple[Document, str]] = []

    for index, document in enumerate(documents, start=1):
        target = destination / document.filename
        if target.exists() and target.stat().st_size > 0 and not force:
            LOG.info(
                "[%2d/%d] %-14s cached (%.0f KB)",
                index,
                len(documents),
                document.doc_id,
                target.stat().st_size / 1024,
            )
            downloaded.append((document, target))
            continue

        try:
            response = session.get(document.url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"response is not a PDF (starts with {response.content[:12]!r})")
            target.write_bytes(response.content)
            LOG.info(
                "[%2d/%d] %-14s downloaded (%.0f KB)",
                index,
                len(documents),
                document.doc_id,
                len(response.content) / 1024,
            )
            downloaded.append((document, target))
        except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the corpus
            LOG.warning("[%2d/%d] %-14s FAILED: %s", index, len(documents), document.doc_id, exc)
            failures.append((document, str(exc)))

        time.sleep(delay_seconds)

    LOG.info("corpus %s: %d downloaded, %d failed", name, len(downloaded), len(failures))
    if failures and not downloaded:
        raise RuntimeError(f"every download failed for corpus {name!r}")
    return downloaded
