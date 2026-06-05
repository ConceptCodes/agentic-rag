from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from typing_extensions import TypedDict


@dataclass(frozen=True, slots=True)
class Citation:
    """A parsed inline citation reference."""

    source: str
    chunk_id: str

    @property
    def text(self) -> str:
        """Return this citation in the answer citation format."""
        return format_citation(self.source, self.chunk_id)


class CitationItem(TypedDict):
    """A retrieval result citation with optional evidence metadata."""

    source: str
    chunk_id: str
    score: float | None
    snippet: str


CITATION_PATTERN_EXAMPLE: Final = "[source: doc_id/chunk_id]"
CITATION_RE: Final = re.compile(r"\[source:\s*(?P<source>[\w./\-]+)/(?P<chunk_id>[\w.\-:]+)\]")


def format_citation(source: str, chunk_id: str) -> str:
    """Format a source and chunk ID as an inline answer citation."""
    return f"[source: {source}/{chunk_id}]"


def parse_citations(text: str) -> list[Citation]:
    """Parse inline answer citations from text."""
    return [
        Citation(source=match.group("source"), chunk_id=match.group("chunk_id"))
        for match in CITATION_RE.finditer(text)
    ]


def extract_citation_strings(text: str) -> list[str]:
    """Extract unique citation strings from text in stable sorted order."""
    return sorted({citation.text for citation in parse_citations(text)})
