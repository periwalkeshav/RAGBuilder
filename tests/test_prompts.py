"""Prompt construction, citation parsing, refusal detection and language routing.

The system prompt is load-bearing: it is the only thing standing between a 7B
model and a confident fabrication. These tests assert the instructions that
matter are actually present, and that the context budget is enforced.
"""

from __future__ import annotations

import pytest

from ragbuilder.llm.prompts import (
    REFUSAL_DE,
    REFUSAL_EN,
    build_qa_prompt,
    build_system_prompt,
    detect_language,
    extract_citations,
    is_refusal,
)
from ragbuilder.retrieval.fusion import RetrievedChunk


def source(text: str, title: str = "BUrlG", section: str = "§ 3", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c",
        text=text,
        doc_id="burlg",
        title=title,
        section=section,
        page_start=page,
        page_end=page,
    )


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "Wie viele Urlaubstage stehen mir zu?",
            "Was ist die gesetzliche Kündigungsfrist?",
            "Welche Merkmale schützt das AGG?",
        ],
    )
    def test_german_questions(self, question):
        assert detect_language(question) == "German"

    @pytest.mark.parametrize(
        "question",
        [
            "What is the statutory minimum annual leave?",
            "How long is the notice period for employees?",
        ],
    )
    def test_english_questions(self, question):
        assert detect_language(question) == "English"

    def test_umlauts_and_section_sign_are_strong_signals(self):
        assert detect_language("§ 622 Kündigungsfrist") == "German"


class TestSystemPrompt:
    def test_instructs_the_model_to_stay_in_the_context(self):
        prompt = build_system_prompt("de", "Wie viele Urlaubstage?")
        assert "ONLY" in prompt
        assert "prior knowledge" in prompt

    def test_requires_citations(self):
        prompt = build_system_prompt("de", "Wie viele Urlaubstage?")
        assert "[1]" in prompt and "cite" in prompt.lower()

    def test_contains_an_explicit_refusal_instruction(self):
        assert REFUSAL_DE in build_system_prompt("de", "Frage")
        assert REFUSAL_EN in build_system_prompt("en", "Question")

    def test_tells_the_model_to_ignore_irrelevant_sources(self):
        """Retrieval always returns top_k, relevant or not.

        Without this, a 7B model treats one off-topic source as evidence that it
        does not understand the question and refuses on a question it can answer.
        """
        prompt = build_system_prompt("de", "Frage")
        assert "not only relevant" in prompt
        assert "Do not refuse merely because" in prompt

    def test_handles_a_general_rule_plus_an_exception(self):
        prompt = build_system_prompt("de", "Frage")
        assert "general rule first" in prompt
        assert "not a reason to refuse" in prompt

    def test_defends_against_prompt_injection(self):
        """Retrieved documents are untrusted input once users can upload PDFs."""
        prompt = build_system_prompt("de", "Frage")
        assert "<context>" in prompt
        assert "ignore" in prompt.lower()
        assert "instruction" in prompt.lower()

    def test_auto_language_follows_the_question(self):
        assert "German" in build_system_prompt("auto", "Wie viele Urlaubstage stehen mir zu?")
        assert "English" in build_system_prompt("auto", "What is the minimum annual leave?")

    def test_explicit_language_overrides_detection(self):
        assert "English" in build_system_prompt("en", "Wie viele Urlaubstage stehen mir zu?")


class TestQaPrompt:
    def test_numbers_the_sources_from_one(self):
        prompt = build_qa_prompt("Frage?", [source("Erster Text"), source("Zweiter Text")])
        assert "[1]" in prompt and "[2]" in prompt
        assert "[0]" not in prompt

    def test_includes_the_citation_line_for_each_source(self):
        prompt = build_qa_prompt("Frage?", [source("Text", title="ArbZG", section="§ 3", page=7)])
        assert "ArbZG, § 3, p. 7" in prompt

    def test_wraps_sources_in_context_delimiters(self):
        prompt = build_qa_prompt("Frage?", [source("Text")])
        assert "<context>" in prompt and "</context>" in prompt

    def test_the_question_appears_after_the_context(self):
        prompt = build_qa_prompt("Wie viele Urlaubstage?", [source("Text")])
        assert prompt.index("</context>") < prompt.index("Wie viele Urlaubstage?")

    def test_history_is_included_when_given(self):
        class Turn:
            question, answer = "Erste Frage", "Erste Antwort"

        prompt = build_qa_prompt("Und danach?", [source("Text")], history=[Turn()])
        assert "Erste Frage" in prompt and "Erste Antwort" in prompt

    def test_history_is_marked_as_not_a_source(self):
        class Turn:
            question, answer = "Q", "A"

        prompt = build_qa_prompt("Frage?", [source("Text")], history=[Turn()])
        assert "do not treat as a source" in prompt

    def test_context_budget_is_enforced(self):
        long_sources = [source("wort " * 5000) for _ in range(5)]
        prompt = build_qa_prompt("Frage?", long_sources, max_context_chars=1000)
        # Budget plus the template scaffolding; the point is it does not run away.
        assert len(prompt) < 3000

    def test_truncation_is_marked(self):
        prompt = build_qa_prompt("Frage?", [source("wort " * 5000)], max_context_chars=500)
        assert "[…]" in prompt

    def test_no_sources_still_renders(self):
        assert "Frage?" in build_qa_prompt("Frage?", [])


class TestCitationExtraction:
    def test_finds_single_citations(self):
        assert extract_citations("Der Urlaub beträgt 24 Werktage [1].", 3) == [1]

    def test_finds_multiple_and_deduplicates(self):
        assert extract_citations("Text [1] mehr [2] und wieder [1].", 3) == [1, 2]

    def test_ignores_out_of_range_numbers(self):
        """A model that invents [9] with 3 sources must not create a phantom citation."""
        assert extract_citations("Laut [9] gilt dies.", 3) == []

    def test_handles_adjacent_citations(self):
        assert extract_citations("Dies gilt [1][2][3].", 3) == [1, 2, 3]

    def test_no_citations(self):
        assert extract_citations("Ein Text ohne Belege.", 3) == []

    def test_zero_is_not_a_valid_citation(self):
        assert extract_citations("Siehe [0].", 3) == []

    def test_handles_comma_separated_numbers_in_one_bracket(self):
        assert extract_citations("Dies gilt [1, 2].", 3) == [1, 2]

    def test_handles_a_paragraph_reference_inside_the_bracket(self):
        """What the model actually writes when told to quote the paragraph.

        Only the leading number is a source reference - harvesting every integer
        would invent a citation to source 3 from the text "§ 3 Abs. 1".
        """
        assert extract_citations("Jährlich mindestens 24 Werktage [1, § 3 Abs. 1].", 5) == [1]

    def test_does_not_invent_citations_from_prose_in_brackets(self):
        assert extract_citations("Siehe [§ 3 Absatz 2] dort.", 5) == []


class TestRefusalDetection:
    def test_detects_the_german_refusal(self):
        assert is_refusal(REFUSAL_DE)

    def test_detects_the_english_refusal(self):
        assert is_refusal(REFUSAL_EN)

    @pytest.mark.parametrize(
        "text",
        [
            "I don't know based on the sources provided.",
            "Ich weiß es nicht.",
            "Die Quellen enthalten keine Information dazu.",
            "The sources do not contain this information.",
        ],
    )
    def test_detects_paraphrases(self, text):
        assert is_refusal(text)

    def test_a_real_answer_is_not_a_refusal(self):
        assert not is_refusal("Der Urlaub beträgt jährlich mindestens 24 Werktage [1].")

    def test_is_case_insensitive(self):
        assert is_refusal("ICH WEISS ES NICHT")
