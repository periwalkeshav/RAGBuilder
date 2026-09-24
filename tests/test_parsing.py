"""PDF parsing helpers: section detection, noise stripping, table rendering.

The pure functions are tested directly; ``parse_pdf`` itself is covered by the
integration suite against the real statute PDFs.
"""

from __future__ import annotations

import pytest

from ragbuilder.ingestion.corpora import CORPORA, get_corpus
from ragbuilder.ingestion.parser import (
    clean_text,
    detect_section,
    is_noise,
    table_to_markdown,
)


class TestSectionDetection:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("§ 3 Mindesturlaub", "§ 3 Mindesturlaub"),
            ("§ 12a Sonderregelung", "§ 12a Sonderregelung"),
            ("§ 5", "§ 5"),
            ("Artikel 12 Berufsfreiheit", "Artikel 12 Berufsfreiheit"),
        ],
    )
    def test_recognises_statute_headings(self, line, expected):
        assert detect_section(line) == expected

    def test_recognises_chapter_headings(self):
        assert detect_section("Abschnitt II Allgemeine Vorschriften").startswith("Abschnitt II")

    def test_normalises_whitespace_in_the_label(self):
        assert detect_section("§   7    Urlaubsgewährung") == "§ 7 Urlaubsgewährung"

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "Der Urlaub beträgt jährlich mindestens 24 Werktage und wird gewährt.",
        ],
    )
    def test_ignores_body_text(self, line):
        assert detect_section(line) is None

    def test_ignores_very_long_lines(self):
        assert detect_section("§ 3 " + "wort " * 60) is None


class TestNoiseFiltering:
    @pytest.mark.parametrize(
        "line",
        [
            "- Seite 3 von 12 -",
            "www.gesetze-im-internet.de",
            "Ein Service des Bundesministeriums der Justiz sowie des Bundesamts für Justiz",
            "42",
        ],
    )
    def test_strips_page_furniture(self, line):
        assert is_noise(line)

    @pytest.mark.parametrize(
        "line",
        [
            "§ 3 Mindesturlaub",
            "Der Urlaub beträgt 24 Werktage.",
        ],
    )
    def test_keeps_real_content(self, line):
        assert not is_noise(line)


class TestCleanText:
    def test_removes_noise_and_returns_the_first_section(self):
        raw = (
            "- Seite 1 von 4 -\n"
            "§ 3 Mindesturlaub\n"
            "Der Urlaub beträgt jährlich mindestens 24 Werktage.\n"
            "www.gesetze-im-internet.de\n"
        )
        text, section = clean_text(raw)
        assert "Seite 1 von 4" not in text
        assert "gesetze-im-internet" not in text
        assert "24 Werktage" in text
        assert section == "§ 3 Mindesturlaub"

    def test_collapses_blank_line_runs_but_keeps_paragraphs(self):
        text, _ = clean_text("Absatz eins.\n\n\n\n\nAbsatz zwei.")
        assert "\n\n\n" not in text
        assert "\n\n" in text

    def test_collapses_extraction_whitespace(self):
        text, _ = clean_text("Der    Urlaub     beträgt")
        assert "    " not in text

    def test_empty_input(self):
        assert clean_text("") == ("", None)


class TestTableToMarkdown:
    def test_renders_a_header_and_separator(self):
        markdown = table_to_markdown([["Jahr", "Tage"], ["2026", "24"]])
        lines = markdown.splitlines()
        assert lines[0] == "| Jahr | Tage |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 2026 | 24 |"

    def test_pads_ragged_rows(self):
        markdown = table_to_markdown([["A", "B", "C"], ["1"]])
        assert markdown.splitlines()[2].count("|") == 4

    def test_handles_none_cells(self):
        assert "| A |  |" in table_to_markdown([["A", None], ["1", "2"]])

    def test_flattens_newlines_inside_cells(self):
        markdown = table_to_markdown([["Kopf\nzeile", "B"], ["1", "2"]])
        assert "Kopf zeile" in markdown
        assert markdown.splitlines()[0].count("|") == 3

    def test_empty_table(self):
        assert table_to_markdown([]) == ""
        assert table_to_markdown([[], [None]]) == ""


class TestCorpora:
    def test_the_default_corpus_is_registered(self):
        assert "german_labour_law" in CORPORA

    def test_the_corpus_is_large_enough_to_be_interesting(self):
        # The project guide asks for 20-50 documents.
        assert 20 <= len(get_corpus("german_labour_law")) <= 50

    def test_document_ids_are_unique(self):
        documents = get_corpus("german_labour_law")
        assert len({d.doc_id for d in documents}) == len(documents)

    def test_filenames_are_unique(self):
        documents = get_corpus("german_labour_law")
        assert len({d.filename for d in documents}) == len(documents)

    def test_every_url_points_at_the_official_source(self):
        for document in get_corpus("german_labour_law"):
            assert document.url.startswith("https://www.gesetze-im-internet.de/")
            assert document.url.endswith(".pdf")

    def test_every_document_is_categorised(self):
        assert all(d.category for d in get_corpus("german_labour_law"))

    def test_unknown_corpus_is_rejected(self):
        with pytest.raises(KeyError, match="unknown corpus"):
            get_corpus("nonexistent")
