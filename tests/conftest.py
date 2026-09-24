"""Shared fixtures.

Nothing here touches PostgreSQL, Qdrant or Ollama - the unit suite has to run in
CI in seconds without a stack behind it. The pieces that need real services live
in ``test_integration.py`` behind ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragbuilder.config import Config  # noqa: E402


@pytest.fixture
def config() -> Config:
    """Defaults with the small embedding model, so nothing large is downloaded."""
    cfg = Config()
    cfg.embedding.model = "intfloat/multilingual-e5-small"
    cfg.embedding.dimensions = 384
    return cfg


@pytest.fixture
def sample_pages() -> list[dict]:
    """Two pages of statute-shaped German text with real section headings."""
    return [
        {
            "page_number": 1,
            "section": "§ 1 Geltungsbereich",
            "has_tables": False,
            "text": (
                "§ 1 Geltungsbereich\n"
                "Dieses Gesetz gilt für Arbeitnehmerinnen und Arbeitnehmer. "
                "Arbeitnehmer im Sinne dieses Gesetzes sind Arbeiter und Angestellte. "
                "Auszubildende gelten ebenfalls als Arbeitnehmer. "
                "Die Vorschriften dieses Abschnitts finden keine Anwendung auf leitende Angestellte."
            ),
        },
        {
            "page_number": 2,
            "section": "§ 3 Mindesturlaub",
            "has_tables": False,
            "text": (
                "§ 3 Mindesturlaub\n"
                "Der Urlaub beträgt jährlich mindestens 24 Werktage. "
                "Als Werktage gelten alle Kalendertage, die nicht Sonn- oder gesetzliche Feiertage sind. "
                "Der Anspruch auf Urlaub entsteht erstmalig nach sechsmonatigem Bestehen des "
                "Arbeitsverhältnisses. "
                "Bei einer Fünftagewoche verringert sich der Anspruch entsprechend."
            ),
        },
    ]


@pytest.fixture
def long_pages() -> list[dict]:
    """Enough text that every chunker produces several chunks."""
    sentence = (
        "Der Arbeitgeber hat die erforderlichen Maßnahmen des Arbeitsschutzes zu treffen. "
        "Die Maßnahmen sind auf ihre Wirksamkeit zu überprüfen und anzupassen. "
    )
    return [
        {
            "page_number": page,
            "section": f"§ {page} Vorschrift",
            "has_tables": False,
            "text": f"§ {page} Vorschrift\n" + sentence * 25,
        }
        for page in range(1, 5)
    ]
