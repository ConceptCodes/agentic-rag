from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from agentic_rag.citations import format_citation
from agentic_rag.constants import CHROMA_DIR, CHUNKS_PATH, DATA_DIR, INDEX_DIR
from agentic_rag.embeddings import get_embedding_adapter

COLLECTION_NAME = "agentic_rag_chunks"


@dataclass(slots=True)
class ChunkRecord:
    """A single chunk of text extracted from a source document."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    position: int
    text: str


def ensure_corpus_directories() -> None:
    """Create project data directories needed by the indexed corpus."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_chroma_client() -> Any:
    """Return a persistent Chroma client rooted at the project chroma directory."""
    import chromadb

    ensure_corpus_directories()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def save_chunk_records(records: list[ChunkRecord]) -> None:
    """Save chunk records to the JSONL chunks file."""
    ensure_corpus_directories()
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def load_chunk_records() -> list[ChunkRecord]:
    """Load chunk records from the JSONL chunks file."""
    if not CHUNKS_PATH.exists():
        return []
    records: list[ChunkRecord] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(ChunkRecord(**row))
    return records


def load_chunk_map() -> dict[str, ChunkRecord]:
    """Return a mapping of chunk_id to chunk record for fast lookup."""
    return {record.chunk_id: record for record in load_chunk_records()}


def index_chunk_records(
    chunks: list[ChunkRecord],
    embedding_backend: str | None = None,
) -> None:
    """Write chunk records to the corpus sidecar and sentence-level Chroma collection."""
    save_chunk_records(chunks)

    adapter = get_embedding_adapter(embedding_backend)
    sentence_records = [
        (chunk, sentence_index, sentence)
        for chunk in chunks
        for sentence_index, sentence in enumerate(_split_sentences(chunk.text))
    ]
    vectors = (
        adapter.embed_documents([sentence for _, _, sentence in sentence_records])
        if sentence_records
        else []
    )

    client = get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    existing = collection.get(include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    if sentence_records:
        collection.add(
            ids=[
                _sentence_id(chunk.chunk_id, sentence_index)
                for chunk, sentence_index, _ in sentence_records
            ],
            documents=[sentence for _, _, sentence in sentence_records],
            embeddings=vectors,
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "position": chunk.position,
                    "sentence_index": sentence_index,
                }
                for chunk, sentence_index, _ in sentence_records
            ],
        )


def keyword_search_chunks(keywords: list[str] | str, top_k: int) -> list[dict[str, Any]]:
    """Return keyword-ranked chunks from the indexed corpus."""
    tokens = _normalize_keywords(keywords)
    chunk_map = load_chunk_map()
    scored: list[tuple[str, float]] = []
    for chunk_id, chunk in chunk_map.items():
        score = 0.0
        text_lower = chunk.text.lower()
        for token in tokens:
            score += float(text_lower.count(token) * max(len(token), 1))
        if score > 0:
            scored.append((chunk_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    hits: list[dict[str, Any]] = []
    for chunk_id, score in scored[:top_k]:
        chunk = chunk_map[chunk_id]
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "score": score,
                "snippet": _keyword_snippet(chunk.text, tokens),
                "citation": format_citation(chunk.source, chunk.chunk_id),
            }
        )
    return hits


def semantic_search_chunks(
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return semantic vector matches from the indexed corpus."""
    adapter = get_embedding_adapter()
    query_vector = adapter.embed_query(query)

    client = get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = response.get("documents", [[]])[0]
    metas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    by_chunk: dict[str, dict[str, Any]] = {}
    for text, meta, distance in zip(docs, metas, distances, strict=True):
        meta = meta or {}
        chunk_id = str(meta.get("chunk_id", "unknown"))
        source = str((meta or {}).get("source", "unknown"))
        score = 1 - float(distance) if distance is not None else None
        current = by_chunk.get(chunk_id)
        if current is None:
            by_chunk[chunk_id] = {
                "chunk_id": chunk_id,
                "source": source,
                "title": meta.get("title", ""),
                "score": score,
                "matched_sentences": [text],
                "snippet": text,
                "citation": format_citation(source, str(chunk_id)),
            }
            continue

        current["matched_sentences"].append(text)
        current["snippet"] = " ... ".join(current["matched_sentences"])
        if score is not None and (current["score"] is None or score > current["score"]):
            current["score"] = score
    return sorted(
        by_chunk.values(),
        key=lambda result: float(result["score"] or 0),
        reverse=True,
    )[:top_k]


def resolve_chunk_ids(chunk_ids: list[str], include_adjacent: bool = False) -> list[str]:
    """Resolve requested chunk IDs to existing IDs, optionally including neighbors."""
    records = load_chunk_records()
    by_id = {record.chunk_id: record for record in records}
    by_doc_pos = {(record.doc_id, record.position): record for record in records}

    selected: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id in by_id:
            selected.append(chunk_id)
            if include_adjacent:
                record = by_id[chunk_id]
                prev_key = (record.doc_id, record.position - 1)
                next_key = (record.doc_id, record.position + 1)
                if prev_key in by_doc_pos:
                    selected.append(by_doc_pos[prev_key].chunk_id)
                if next_key in by_doc_pos:
                    selected.append(by_doc_pos[next_key].chunk_id)

    return list(dict.fromkeys(selected))


def read_chunks(chunk_ids: list[str], include_adjacent: bool = False) -> list[dict[str, Any]]:
    """Return full chunk records by ID, optionally including neighbors."""
    records = load_chunk_records()
    by_id = {record.chunk_id: record for record in records}
    deduped = resolve_chunk_ids(chunk_ids, include_adjacent=include_adjacent)
    results: list[dict[str, Any]] = []
    for chunk_id in deduped:
        record = by_id[chunk_id]
        results.append(
            {
                "chunk_id": record.chunk_id,
                "source": record.source,
                "title": record.title,
                "text": record.text,
                "citation": format_citation(record.source, record.chunk_id),
            }
        )
    return results


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _normalize_keywords(keywords: list[str] | str) -> list[str]:
    if isinstance(keywords, str):
        return _tokenize(keywords)
    return [keyword.strip().lower() for keyword in keywords if keyword.strip()]


def _keyword_snippet(text: str, keywords: list[str], max_sentences: int = 3) -> str:
    matches = [
        sentence
        for sentence in _split_sentences(text)
        if any(keyword in sentence.lower() for keyword in keywords)
    ]
    if not matches:
        return text[:220]
    return " ... ".join(matches[:max_sentences])


def _split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()
    ]
    return sentences if sentences else [text.strip()]


def _sentence_id(chunk_id: str, sentence_index: int) -> str:
    return f"{chunk_id}:sent:{sentence_index}"
