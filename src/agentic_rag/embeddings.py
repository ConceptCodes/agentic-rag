from __future__ import annotations

import os
from typing import Protocol

from agentic_rag.constants import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL_LMSTUDIO,
    DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS,
    LMSTUDIO_BASE_URL,
)


class EmbeddingAdapter(Protocol):
    """Protocol for embedding backends that produce vector embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a vector."""


class SentenceTransformerAdapter:
    """Embedding backend using sentence-transformers with local model inference."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into normalized vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

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
        from langchain_openai import OpenAIEmbeddings

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


def get_embedding_adapter(
    embedding_backend: str | None = None,
) -> EmbeddingAdapter:
    """Resolve and return the embedding adapter for the requested backend."""
    backend = (embedding_backend or DEFAULT_EMBEDDING_BACKEND).lower()
    if backend == "lmstudio":
        return LMStudioEmbeddingAdapter()
    return SentenceTransformerAdapter()
