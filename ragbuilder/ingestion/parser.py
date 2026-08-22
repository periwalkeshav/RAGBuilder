"""PDF parsing with PyMuPDF.

Three things this does that a naive ``page.get_text()`` loop does not:

1. **Keeps page numbers.** Every citation the RAG chain emits points at a page,
   so the page boundary has to survive parsing. Chunks carry ``page_start`` and
   ``page_end`` all the way to the answer.
2. **Detects section headings.** German statutes are organised into ``§ n``
   paragraphs; the section a chunk belongs to is worth more than its offset in
   the file, both for citation quality and for chunking that respects meaning.
3. **Converts tables to Markdown.** PyMuPDF's table finder returns cell grids;
   flattened into prose they become word salad, and as Markdown they stay
   readable to the embedding model and to the LLM.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

LOG = logging.getLogger("ragbuilder.parser")

# "§ 1", "§ 12a", "§§ 3 bis 5" - the backbone of a German statute.
SECTION_PATTERNS = (
    re.compile(r"^\s*(§+\s*\d+[a-z]?(?:\s*[-–bis]+\s*\d+[a-z]?)?)\s*(.*)$"),
    re.compile(r"^\s*(Artikel\s+\d+[a-z]?)\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*(Abschnitt\s+[IVXLC\d]+)\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*(Kapitel\s+[IVXLC\d]+)\s*(.*)$", re.IGNORECASE),
    # English fallback for the arXiv corpus.
    re.compile(r"^\s*(\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z ]{3,60})\s*$"),
)

# Page furniture that adds nothing and dilutes embeddings.
NOISE_PATTERNS = (
    re.compile(r"^\s*-\s*Seite\s+\d+\s+von\s+\d+\s*-\s*$", re.IGNORECASE),
    re.compile(r"^\s*Ein Service des Bundesministeriums der Justiz.*$", re.IGNORECASE),
    re.compile(r"^\s*www\.gesetze-im-internet\.de\s*$", re.IGNORECASE),
    re.compile(r"^\s*Seite \d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # bare page numbers
)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    section: str | None = None
    has_tables: bool = False


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    content_hash: str
    pages: list[ParsedPage] = field(default_factory=list)
    table_count: int = 0
    title_from_pdf: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def detect_section(line: str) -> str | None:
    """Return a normalised section label if ``line`` looks like a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    for pattern in SECTION_PATTERNS:
        match = pattern.match(stripped)
        if match:
            label = re.sub(r"\s+", " ", match.group(1)).strip()
            heading = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
            return f"{label} {heading}".strip() if heading else label
    return None


def clean_text(raw: str) -> tuple[str, str | None]:
    """Strip page furniture and return ``(text, first section seen)``."""
    kept: list[str] = []
    first_section: str | None = None

    for line in raw.splitlines():
        if is_noise(line):
            continue
        section = detect_section(line)
        if section and first_section is None:
            first_section = section
        kept.append(line.rstrip())

    text = "\n".join(kept)
    # Collapse the runs of blank lines PDF extraction loves to produce, but keep
    # paragraph breaks - the sentence chunker uses them.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), first_section


def table_to_markdown(table_rows: list[list[str | None]]) -> str:
    """Render a PyMuPDF table grid as a Markdown table."""
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table_rows if row]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def parse_pdf(path: Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse ``path`` into pages with sections, tables and page numbers intact."""
    doc_id = doc_id or path.stem.lower()
    document = fitz.open(path)

    parsed = ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        content_hash=file_hash(path),
        title_from_pdf=(document.metadata or {}).get("title") or None,
    )

    current_section: str | None = None

    try:
        for index, page in enumerate(document, start=1):
            raw = page.get_text("text")
            text, section = clean_text(raw)

            tables_markdown: list[str] = []
            try:
                finder = page.find_tables()
                for table in finder.tables:
                    markdown = table_to_markdown(table.extract())
                    if markdown:
                        tables_markdown.append(markdown)
            except Exception as exc:  # noqa: BLE001 - table finding is best effort
                LOG.debug("table extraction failed on %s p%d: %s", path.name, index, exc)

            if tables_markdown:
                parsed.table_count += len(tables_markdown)
                text = f"{text}\n\n" + "\n\n".join(tables_markdown)

            # A page with no heading of its own continues the previous section.
            current_section = section or current_section

            if text.strip():
                parsed.pages.append(
                    ParsedPage(
                        page_number=index,
                        text=text,
                        section=current_section,
                        has_tables=bool(tables_markdown),
                    )
                )
    finally:
        document.close()

    LOG.info(
        "parsed %-18s %3d pages, %6d chars, %d table(s)",
        path.name,
        parsed.page_count,
        parsed.char_count,
        parsed.table_count,
    )
    return parsed
