from __future__ import annotations

from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from agentic_rag.constants import DEFAULT_TOP_K
from agentic_rag.context_tracker import ContextTracker
from agentic_rag.corpus import (
    keyword_search_chunks,
    read_chunks,
    resolve_chunk_ids,
    semantic_search_chunks,
)

CONTEXT_TRACKER = ContextTracker()


def reset_context_tracker() -> None:
    """Reset the read-chunk tracker for a new agent invocation."""
    CONTEXT_TRACKER.read_chunk_ids.clear()


class KeywordSearchInput(BaseModel):
    """Input schema for keyword-based chunk retrieval."""

    keywords: list[str] = Field(description="Short, specific keywords or entity names to match")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@tool("keyword_search", args_schema=KeywordSearchInput)
def keyword_search(keywords: list[str], top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Keyword-level retrieval over chunk text. Use this for exact entities and precise terms."""
    return {
        "tool": "keyword_search",
        "keywords": keywords,
        "results": keyword_search_chunks(keywords, top_k),
    }


class SemanticSearchInput(BaseModel):
    """Input schema for semantic vector-based chunk retrieval."""

    query: str = Field(description="Semantic query string describing concepts or intent")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@tool("semantic_search", args_schema=SemanticSearchInput)
def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Semantic retrieval over vector index. Use for synonymy and concept matching."""
    return {
        "tool": "semantic_search",
        "query": query,
        "results": semantic_search_chunks(query, top_k),
    }


class ChunkReadInput(BaseModel):
    """Input schema for reading full chunk content by ID."""

    chunk_ids: list[str] = Field(description="Chunk IDs to read in full")
    include_adjacent: bool = Field(
        default=False, description="Include previous and next chunks if present"
    )


@tool("chunk_read", args_schema=ChunkReadInput)
def chunk_read(chunk_ids: list[str], include_adjacent: bool = False) -> dict[str, Any]:
    """Read full chunk content when snippets are not sufficient for answering."""
    resolved_ids = resolve_chunk_ids(chunk_ids, include_adjacent=include_adjacent)
    unread_ids, repeated_ids = CONTEXT_TRACKER.split_unread(resolved_ids)
    results = read_chunks(unread_ids, include_adjacent=False)
    CONTEXT_TRACKER.mark_read([str(result["chunk_id"]) for result in results])
    return {
        "tool": "chunk_read",
        "requested": chunk_ids,
        "include_adjacent": include_adjacent,
        "results": results,
        "skipped": [
            {
                "chunk_id": chunk_id,
                "reason": "already_read",
                "message": "Chunk was already read in this retrieval episode.",
            }
            for chunk_id in repeated_ids
        ],
    }


ALL_TOOLS = [keyword_search, semantic_search, chunk_read]
