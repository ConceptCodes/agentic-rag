from __future__ import annotations

import json
import re
from dataclasses import asdict

from agentic_rag.ingest import split_text
from agentic_rag.utils import ChunkRecord


def test_split_text_basic() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_text(text, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_sentence_boundary() -> None:
    text = "A. B. C."
    chunks = split_text(text, max_tokens=2)
    assert len(chunks) >= 2


def test_chunk_record_roundtrip() -> None:
    record = ChunkRecord(
        chunk_id="test:0",
        doc_id="test",
        source="test.md",
        title="Test",
        position=0,
        text="Hello world",
    )
    data = asdict(record)
    restored = ChunkRecord(**data)
    assert restored == record


def test_chunk_record_jsonl() -> None:
    record = ChunkRecord(
        chunk_id="jsonl:0",
        doc_id="jsonl",
        source="test.jsonl",
        title="JSONL",
        position=0,
        text="Roundtrip test",
    )
    line = json.dumps(asdict(record), ensure_ascii=True)
    restored = ChunkRecord(**json.loads(line))
    assert restored == record


def test_citation_regex_valid() -> None:
    pattern = re.compile(r"\[source:\s*[\w./\-]+/[\w.\-:]+\]")
    valid = "[source: company_docs/policy.md/550e8400-e29b-41d4-a716-446655440000:3]"
    assert pattern.search(valid)


def test_citation_regex_invalid_no_bracket() -> None:
    pattern = re.compile(r"\[source:\s*[\w./\-]+/[\w.\-:]+\]")
    invalid = "source: foo/bar"
    assert not pattern.search(invalid)


def test_citation_regex_invalid_wrong_prefix() -> None:
    pattern = re.compile(r"\[source:\s*[\w./\-]+/[\w.\-:]+\]")
    invalid = "[other: foo/bar]"
    assert not pattern.search(invalid)
