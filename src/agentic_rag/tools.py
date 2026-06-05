from __future__ import annotations

import re
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from agentic_rag.constants import DEFAULT_TOP_K
from agentic_rag.utils import (
    get_chroma_client,
    get_embedding_adapter,
    load_chunk_map,
    load_chunk_records,
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]


def _snippet(text: str, query: str, max_chars: int = 220) -> str:
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:max_chars]
    start = max(0, idx - max_chars // 3)
    end = min(len(text), idx + max_chars)
    return text[start:end]


class KeywordSearchInput(BaseModel):
    """Input schema for keyword-based chunk retrieval."""

    query: str = Field(description="Lexical query string with specific entities or terms")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@tool("keyword_search", args_schema=KeywordSearchInput)
def keyword_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Keyword-level retrieval over chunk text. Use this for exact entities and precise terms."""
    tokens = _tokenize(query)
    chunk_map = load_chunk_map()
    scored: list[tuple[str, float]] = []
    for chunk_id, chunk in chunk_map.items():
        score = 0.0
        text_lower = chunk.text.lower()
        for token in tokens:
            score += float(text_lower.count(token) * max(len(token), 1))
        if score > 0:
            scored.append((chunk_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    hits = []
    for chunk_id, score in scored[:top_k]:
        chunk = chunk_map[chunk_id]
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "score": score,
                "snippet": _snippet(chunk.text, query),
                "citation": f"[source: {chunk.source}/{chunk.chunk_id}]",
            }
        )
    return {"tool": "keyword_search", "query": query, "results": hits}


class SemanticSearchInput(BaseModel):
    """Input schema for semantic vector-based chunk retrieval."""

    query: str = Field(description="Semantic query string describing concepts or intent")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@tool("semantic_search", args_schema=SemanticSearchInput)
def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Semantic retrieval over vector index. Use for synonymy and concept matching."""
    adapter = get_embedding_adapter()
    query_vector = adapter.embed_query(query)

    client = get_chroma_client()
    collection = client.get_or_create_collection("agentic_rag_chunks")
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    docs = response.get("documents", [[]])[0]
    metas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    results = []
    for chunk_id, text, meta, distance in zip(ids, docs, metas, distances, strict=True):
        source = str((meta or {}).get("source", "unknown"))
        results.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "title": (meta or {}).get("title", ""),
                "score": 1 - float(distance) if distance is not None else None,
                "snippet": _snippet(text, query),
                "citation": f"[source: {source}/{chunk_id}]",
            }
        )

    return {"tool": "semantic_search", "query": query, "results": results}


class ChunkReadInput(BaseModel):
    """Input schema for reading full chunk content by ID."""

    chunk_ids: list[str] = Field(description="Chunk IDs to read in full")
    include_adjacent: bool = Field(
        default=False, description="Include previous and next chunks if present"
    )


@tool("chunk_read", args_schema=ChunkReadInput)
def chunk_read(chunk_ids: list[str], include_adjacent: bool = False) -> dict[str, Any]:
    """Read full chunk content when snippets are not sufficient for answering."""
    records = load_chunk_records()
    by_id = {r.chunk_id: r for r in records}
    by_doc_pos = {(r.doc_id, r.position): r for r in records}

    selected: list[str] = []
    for cid in chunk_ids:
        if cid in by_id:
            selected.append(cid)
            if include_adjacent:
                rec = by_id[cid]
                prev_key = (rec.doc_id, rec.position - 1)
                next_key = (rec.doc_id, rec.position + 1)
                if prev_key in by_doc_pos:
                    selected.append(by_doc_pos[prev_key].chunk_id)
                if next_key in by_doc_pos:
                    selected.append(by_doc_pos[next_key].chunk_id)

    deduped = list(dict.fromkeys(selected))
    results = []
    for cid in deduped:
        rec = by_id[cid]
        results.append(
            {
                "chunk_id": rec.chunk_id,
                "source": rec.source,
                "title": rec.title,
                "text": rec.text,
                "citation": f"[source: {rec.source}/{rec.chunk_id}]",
            }
        )

    return {
        "tool": "chunk_read",
        "requested": chunk_ids,
        "include_adjacent": include_adjacent,
        "results": results,
    }


ALL_TOOLS = [keyword_search, semantic_search, chunk_read]
