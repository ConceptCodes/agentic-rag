from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from agentic_rag.constants import (
    CHROMA_DIR,
    CHUNKS_PATH,
    DATA_DIR,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL_LMSTUDIO,
    DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS,
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_TOP_K,
    INDEX_DIR,
    LMSTUDIO_BASE_URL,
)


@dataclass(slots=True)
class ChunkRecord:
    """A single chunk of text extracted from a source document."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    position: int
    text: str


class EmbeddingAdapter(Protocol):
    """Protocol for embedding backends that produce vector embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a vector."""


class SentenceTransformerAdapter:
    """Embedding backend using sentence-transformers with local model inference."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into normalized vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a vector."""
        return self.embed_documents([text])[0]


class LMStudioEmbeddingAdapter:
    """Embedding backend using an OpenAI-compatible endpoint (LM Studio)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL_LMSTUDIO,
        base_url: str = LMSTUDIO_BASE_URL,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts via the remote embedding API."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string via the remote embedding API."""
        return self.embeddings.embed_query(text)


def ensure_directories() -> None:
    """Create project data directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    """Extract the configurable dict from a RunnableConfig, or return empty."""
    if not config:
        return {}
    return dict(config.get("configurable") or {})


def get_runtime_settings(config: RunnableConfig | None = None) -> dict[str, Any]:
    """Extract runtime settings from configurable config, filling defaults."""
    cfg = get_configurable(config)
    return {
        "thread_id": str(cfg.get("thread_id", "local-thread")),
        "embedding_backend": str(cfg.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)),
        "top_k": int(cfg.get("top_k", DEFAULT_TOP_K)),
        "max_steps": int(cfg.get("max_steps", DEFAULT_MAX_AGENT_STEPS)),
        "model": str(cfg.get("model", DEFAULT_CHAT_MODEL)),
    }


def get_embedding_adapter(
    config: RunnableConfig | None = None,
) -> EmbeddingAdapter:
    """Resolve and return the embedding adapter based on the backend setting."""
    cfg = get_configurable(config)
    backend = str(cfg.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)).lower()
    if backend == "lmstudio":
        model_name = str(cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL_LMSTUDIO))
        base_url = str(cfg.get("base_url", LMSTUDIO_BASE_URL))
        return LMStudioEmbeddingAdapter(model_name=model_name, base_url=base_url)
    model_name = str(cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS))
    return SentenceTransformerAdapter(model_name=model_name)


def get_chroma_client() -> Any:
    """Return a persistent Chroma client rooted at the project chroma directory."""
    ensure_directories()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def save_chunk_records(records: list[ChunkRecord]) -> None:
    """Save chunk records to the JSONL chunks file."""
    ensure_directories()
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
    """Return a mapping of chunk_id -> ChunkRecord for fast lookup."""
    return {r.chunk_id: r for r in load_chunk_records()}


def resolve_repo_path(path: str) -> Path:
    """Resolve a relative path against the project root, or return absolute as-is."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p
