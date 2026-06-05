from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.constants import COMPANY_DOCS_DIR, DEFAULT_CHUNK_MAX_TOKENS
from src.utils import (
    ChunkRecord,
    get_chroma_client,
    get_embedding_adapter,
    save_chunk_records,
)

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", flags=re.DOTALL)


@dataclass(slots=True)
class MarkdownDocument:
    doc_id: str
    title: str
    source: str
    text: str


def load_markdown_documents(
    docs_dir: Path = COMPANY_DOCS_DIR,
) -> list[MarkdownDocument]:
    docs: list[MarkdownDocument] = []
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name.lower() == "sources.md":
            continue
        raw = path.read_text(encoding="utf-8")
        title, content = _parse_frontmatter(raw, path.stem)
        rel = str(path.relative_to(docs_dir))
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, rel))
        docs.append(
            MarkdownDocument(doc_id=doc_id, title=title, source=rel, text=content)
        )
    return docs


def _parse_frontmatter(raw: str, default_title: str) -> tuple[str, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return default_title, raw.strip()
    body = raw[match.end() :].strip()
    frontmatter = match.group(1)
    title = default_title
    for line in frontmatter.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or default_title
            break
    return title, body


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    try:
        return len(_ENCODER.encode(text))
    except Exception:
        # Keep ingestion resilient if tokenization fails unexpectedly.
        return len(TOKEN_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    return parts if parts else [text.strip()]


def split_text(text: str, max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS) -> list[str]:
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        if sentence_tokens > max_tokens:
            # If one sentence exceeds the limit, split hard by token window.
            try:
                tokens = _ENCODER.encode(sentence)
            except Exception:
                tokens = TOKEN_RE.findall(sentence)

            for i in range(0, len(tokens), max_tokens):
                window_tokens = tokens[i : i + max_tokens]
                try:
                    window = _ENCODER.decode(window_tokens)
                except Exception:
                    window = " ".join(str(t) for t in window_tokens)
                window = window.strip()
                if window:
                    if current:
                        chunks.append(" ".join(current).strip())
                        current = []
                        current_tokens = 0
                    chunks.append(window)
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def build_chunks(docs: list[MarkdownDocument]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for doc in docs:
        for i, chunk in enumerate(split_text(doc.text)):
            chunk_id = f"{doc.doc_id}:{i}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    position=i,
                    text=chunk,
                )
            )
    return records


def index_documents(embedding_backend: str | None = None) -> dict[str, int]:
    docs = load_markdown_documents()
    chunks = build_chunks(docs)
    save_chunk_records(chunks)

    config = (
        {"configurable": {"embedding_backend": embedding_backend}}
        if embedding_backend
        else None
    )
    adapter = get_embedding_adapter(config)
    vectors = adapter.embed_documents([c.text for c in chunks]) if chunks else []

    client = get_chroma_client()
    collection = client.get_or_create_collection("agentic_rag_chunks")
    if chunks:
        # Replace collection contents in a simple way for deterministic local iteration.
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "title": c.title,
                    "position": c.position,
                }
                for c in chunks
            ],
        )

    return {"documents": len(docs), "chunks": len(chunks)}
