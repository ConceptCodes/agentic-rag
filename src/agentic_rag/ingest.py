from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from agentic_rag.constants import COMPANY_DOCS_DIR, DEFAULT_CHUNK_MAX_TOKENS
from agentic_rag.corpus import ChunkRecord, index_chunk_records

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", flags=re.DOTALL)


@dataclass(slots=True)
class MarkdownDocument:
    """A parsed markdown document with frontmatter metadata."""

    doc_id: str
    title: str
    source: str
    text: str


def load_markdown_documents(
    docs_dir: Path = COMPANY_DOCS_DIR,
) -> list[MarkdownDocument]:
    """Load all markdown files from the docs directory into parsed documents."""
    docs: list[MarkdownDocument] = []
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name.lower() == "sources.md":
            continue
        raw = path.read_text(encoding="utf-8")
        title, content = _parse_frontmatter(raw, path.stem)
        rel = str(path.relative_to(docs_dir))
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, rel))
        docs.append(MarkdownDocument(doc_id=doc_id, title=title, source=rel, text=content))
    return docs


def _parse_frontmatter(raw: str, default_title: str) -> tuple[str, str]:
    """Extract YAML frontmatter title from raw markdown content."""
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
    """Count tokens in text using tiktoken, falling back to regex on tokenizer error."""
    try:
        return len(_ENCODER.encode(text))
    except Exception:
        logger.warning("tiktoken encoding failed, falling back to regex tokenizer", exc_info=True)
        return len(TOKEN_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    return parts if parts else [text.strip()]


def split_text(text: str, max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS) -> list[str]:
    """Split text into chunks at sentence boundaries, capped at max_tokens each."""
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
                logger.warning(
                    "tiktoken encode failed for long sentence, falling back to regex",
                    exc_info=True,
                )
                tokens = TOKEN_RE.findall(sentence)

            for i in range(0, len(tokens), max_tokens):
                window_tokens = tokens[i : i + max_tokens]
                try:
                    window = _ENCODER.decode(window_tokens)
                except Exception:
                    logger.warning(
                        "tiktoken decode failed, falling back to concatenation",
                        exc_info=True,
                    )
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
    """Convert parsed documents into a flat list of chunk records."""
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
    """Load, chunk, and index all markdown documents into Chroma and JSONL."""
    docs = load_markdown_documents()
    chunks = build_chunks(docs)
    index_chunk_records(chunks, embedding_backend=embedding_backend)

    return {"documents": len(docs), "chunks": len(chunks)}
