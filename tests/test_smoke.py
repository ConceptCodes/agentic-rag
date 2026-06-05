from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from agentic_rag import corpus
from agentic_rag.agent import (
    _build_model,
    _reasoning_config,
    resolve_model_name,
)
from agentic_rag.citations import (
    Citation,
    extract_citation_strings,
    format_citation,
    parse_citations,
)
from agentic_rag.constants import COMPANY_DOCS_DIR, PROJECT_ROOT
from agentic_rag.corpus import ChunkRecord
from agentic_rag.ingest import split_text
from agentic_rag.tools import chunk_read, reset_context_tracker
from agentic_rag.trace import FinalStep, ToolCallStep, parse_agent_trace
from eval.harness import EvalSample, run_eval, score_answer


def test_split_text_basic() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_text(text, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_company_docs_dir_uses_repo_data_dir() -> None:
    assert COMPANY_DOCS_DIR == PROJECT_ROOT / "data" / "company_docs"
    assert COMPANY_DOCS_DIR.name == "company_docs"


def test_chat_model_aliases_and_defaults() -> None:
    assert resolve_model_name(None, chat_provider="lmstudio") == "google/gemma-4-e4b"
    assert resolve_model_name(None, chat_provider="openrouter") == "deepseek/deepseek-v4-pro"
    assert resolve_model_name("deepseek-v4-pro", chat_provider="openrouter") == (
        "deepseek/deepseek-v4-pro"
    )
    assert resolve_model_name("gpt-5-mini", chat_provider="openrouter") == "openai/gpt-5-mini"


def test_reasoning_config() -> None:
    assert _reasoning_config(False) is None
    assert _reasoning_config(False, "high") is None
    assert _reasoning_config(True) == {"enabled": True}
    assert _reasoning_config(True, "high") == {"enabled": True, "effort": "high"}


def test_openrouter_model_configuration(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Agentic RAG")

    model = _build_model(
        model_name="deepseek/deepseek-v4-pro",
        chat_provider="openrouter",
        reasoning_enabled=True,
        reasoning_effort="xhigh",
    )

    assert str(model.openai_api_base) == "https://openrouter.ai/api/v1"
    assert model.model_name == "deepseek/deepseek-v4-pro"
    assert model.reasoning == {"enabled": True, "effort": "xhigh"}
    assert model.default_headers == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "Agentic RAG",
    }


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


def test_chunk_record_jsonl(tmp_path: Path, monkeypatch) -> None:
    _use_tmp_corpus(monkeypatch, tmp_path)
    record = ChunkRecord(
        chunk_id="jsonl:0",
        doc_id="jsonl",
        source="test.jsonl",
        title="JSONL",
        position=0,
        text="Roundtrip test",
    )
    corpus.save_chunk_records([record])
    assert corpus.load_chunk_records() == [record]


def test_keyword_search_chunks_returns_citations(tmp_path: Path, monkeypatch) -> None:
    _use_tmp_corpus(monkeypatch, tmp_path)
    record = ChunkRecord(
        chunk_id="jsonl:0",
        doc_id="jsonl",
        source="policy.md",
        title="Policy",
        position=0,
        text="Laptop reimbursement applies to remote employees.",
    )
    corpus.save_chunk_records([record])

    results = corpus.keyword_search_chunks("reimbursement", top_k=5)

    assert results == [
        {
            "chunk_id": "jsonl:0",
            "source": "policy.md",
            "title": "Policy",
            "score": 13.0,
            "snippet": "Laptop reimbursement applies to remote employees.",
            "citation": "[source: policy.md/jsonl:0]",
        }
    ]


def test_read_chunks_includes_adjacent_records(tmp_path: Path, monkeypatch) -> None:
    _use_tmp_corpus(monkeypatch, tmp_path)
    records = [
        ChunkRecord("doc:0", "doc", "policy.md", "Policy", 0, "First"),
        ChunkRecord("doc:1", "doc", "policy.md", "Policy", 1, "Second"),
        ChunkRecord("doc:2", "doc", "policy.md", "Policy", 2, "Third"),
    ]
    corpus.save_chunk_records(records)

    results = corpus.read_chunks(["doc:1"], include_adjacent=True)

    assert [result["chunk_id"] for result in results] == ["doc:1", "doc:0", "doc:2"]


def test_chunk_read_context_tracker_skips_repeated_chunks(tmp_path: Path, monkeypatch) -> None:
    _use_tmp_corpus(monkeypatch, tmp_path)
    reset_context_tracker()
    records = [
        ChunkRecord("doc:0", "doc", "policy.md", "Policy", 0, "First"),
        ChunkRecord("doc:1", "doc", "policy.md", "Policy", 1, "Second"),
    ]
    corpus.save_chunk_records(records)

    first = chunk_read.invoke({"chunk_ids": ["doc:0"]})
    second = chunk_read.invoke({"chunk_ids": ["doc:0"]})

    assert [result["chunk_id"] for result in first["results"]] == ["doc:0"]
    assert first["skipped"] == []
    assert second["results"] == []
    assert second["skipped"] == [
        {
            "chunk_id": "doc:0",
            "reason": "already_read",
            "message": "Chunk was already read in this retrieval episode.",
        }
    ]


def test_chunk_read_context_tracker_marks_adjacent_chunks(tmp_path: Path, monkeypatch) -> None:
    _use_tmp_corpus(monkeypatch, tmp_path)
    reset_context_tracker()
    records = [
        ChunkRecord("doc:0", "doc", "policy.md", "Policy", 0, "First"),
        ChunkRecord("doc:1", "doc", "policy.md", "Policy", 1, "Second"),
        ChunkRecord("doc:2", "doc", "policy.md", "Policy", 2, "Third"),
    ]
    corpus.save_chunk_records(records)

    first = chunk_read.invoke({"chunk_ids": ["doc:1"], "include_adjacent": True})
    second = chunk_read.invoke({"chunk_ids": ["doc:0", "doc:2"]})

    assert [result["chunk_id"] for result in first["results"]] == ["doc:1", "doc:0", "doc:2"]
    assert second["results"] == []
    assert [skipped["chunk_id"] for skipped in second["skipped"]] == ["doc:0", "doc:2"]


def test_citation_format_parse_roundtrip() -> None:
    formatted = format_citation("company_docs/policy.md", "550e8400-e29b-41d4-a716-446655440000:3")
    assert formatted == "[source: company_docs/policy.md/550e8400-e29b-41d4-a716-446655440000:3]"
    assert parse_citations(formatted) == [
        Citation(
            source="company_docs/policy.md",
            chunk_id="550e8400-e29b-41d4-a716-446655440000:3",
        )
    ]


def test_citation_parse_invalid_no_bracket() -> None:
    invalid = "source: foo/bar"
    assert parse_citations(invalid) == []


def test_citation_parse_invalid_wrong_prefix() -> None:
    invalid = "[other: foo/bar]"
    assert parse_citations(invalid) == []


def test_extract_citation_strings_deduplicates() -> None:
    citation = format_citation("company_docs/policy.md", "chunk:1")
    assert extract_citation_strings(f"{citation} repeated {citation}") == [citation]


def test_score_answer_contain_match() -> None:
    assert score_answer("Remote", "Remote employees are covered.")
    assert not score_answer("Office", "Remote employees are covered.")


def test_run_eval_reports_hits_and_errors() -> None:
    samples = [
        EvalSample(question="q1", answer="alpha"),
        EvalSample(question="q2", answer="beta"),
        EvalSample(question="q3", answer="gamma"),
    ]

    def agent_fn(question: str, thread_id: str) -> str:
        assert thread_id.startswith("eval-")
        if question == "q3":
            msg = "boom"
            raise RuntimeError(msg)
        return "alpha answer" if question == "q1" else "wrong"

    report = run_eval(samples, agent_fn)

    assert report.samples == 3
    assert report.hits == 1
    assert report.contain_match_accuracy == 1 / 3
    assert report.results[2].error == "boom"
    assert report.summary() == {
        "samples": 3,
        "contain_match_accuracy": 1 / 3,
        "hits": 1,
    }


def test_parse_agent_trace_normalizes_tool_calls() -> None:
    result_raw = {
        "messages": [
            SimpleNamespace(type="human", content="question"),
            SimpleNamespace(
                type="ai",
                content="looking",
                tool_calls=[
                    {
                        "name": "keyword_search",
                        "args": '{"query": "remote", "top_k": 2}',
                        "id": "call-1",
                    }
                ],
            ),
            SimpleNamespace(
                type="tool",
                name="keyword_search",
                content='{"tool": "keyword_search", "results": []}',
            ),
            SimpleNamespace(type="ai", content="Final [source: policy.md/chunk:1]"),
        ]
    }

    steps = parse_agent_trace(result_raw)

    assert isinstance(steps[0], ToolCallStep)
    assert steps[0].thought == "looking"
    assert steps[0].tool_calls[0].name == "keyword_search"
    assert steps[0].tool_calls[0].args == {"query": "remote", "top_k": 2}
    assert steps[0].tool_results[0].parsed == {"tool": "keyword_search", "results": []}
    assert isinstance(steps[1], FinalStep)
    assert steps[1].citations == ["[source: policy.md/chunk:1]"]


def test_parse_agent_trace_handles_openai_function_shape() -> None:
    result_raw = {
        "messages": [
            SimpleNamespace(
                type="ai",
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "function": {
                                "name": "chunk_read",
                                "arguments": '{"chunk_ids": ["chunk:1"]}',
                            },
                        }
                    ]
                },
            )
        ]
    }

    steps = parse_agent_trace(result_raw)

    assert isinstance(steps[0], ToolCallStep)
    assert steps[0].tool_calls[0].name == "chunk_read"
    assert steps[0].tool_calls[0].args == {"chunk_ids": ["chunk:1"]}


def _use_tmp_corpus(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(corpus, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(corpus, "INDEX_DIR", tmp_path / "data" / "index")
    monkeypatch.setattr(corpus, "CHROMA_DIR", tmp_path / "data" / "index" / "chroma")
    monkeypatch.setattr(corpus, "CHUNKS_PATH", tmp_path / "data" / "index" / "chunks.jsonl")
