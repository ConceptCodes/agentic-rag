from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from src.constants import (
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
	chunk_id: str
	doc_id: str
	source: str
	title: str
	position: int
	text: str


class SentenceTransformerAdapter:
	def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS) -> None:
		self.model = SentenceTransformer(model_name)

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		vectors = self.model.encode(texts, normalize_embeddings=True)
		return [v.tolist() for v in vectors]

	def embed_query(self, text: str) -> list[float]:
		return self.embed_documents([text])[0]


class LMStudioEmbeddingAdapter:
	def __init__(
		self,
		model_name: str = DEFAULT_EMBEDDING_MODEL_LMSTUDIO,
		base_url: str = LMSTUDIO_BASE_URL,
	) -> None:
		api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
		self.embeddings = OpenAIEmbeddings(
			model=model_name,
			api_key=api_key,
			base_url=base_url,
		)

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return self.embeddings.embed_documents(texts)

	def embed_query(self, text: str) -> list[float]:
		return self.embeddings.embed_query(text)


def ensure_directories() -> None:
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	INDEX_DIR.mkdir(parents=True, exist_ok=True)
	CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
	if not config:
		return {}
	return dict(config.get("configurable") or {})


def get_runtime_settings(config: RunnableConfig | None = None) -> dict[str, Any]:
	cfg = get_configurable(config)
	return {
		"thread_id": str(cfg.get("thread_id", "local-thread")),
		"embedding_backend": str(cfg.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)),
		"top_k": int(cfg.get("top_k", DEFAULT_TOP_K)),
		"max_steps": int(cfg.get("max_steps", DEFAULT_MAX_AGENT_STEPS)),
		"model": str(cfg.get("model", DEFAULT_CHAT_MODEL)),
	}


def get_embedding_adapter(config: RunnableConfig | None = None):
	cfg = get_configurable(config)
	backend = str(cfg.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)).lower()
	if backend == "lmstudio":
		model_name = str(cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL_LMSTUDIO))
		base_url = str(cfg.get("base_url", LMSTUDIO_BASE_URL))
		return LMStudioEmbeddingAdapter(model_name=model_name, base_url=base_url)
	model_name = str(cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS))
	return SentenceTransformerAdapter(model_name=model_name)


def get_chroma_client() -> chromadb.PersistentClient:
	ensure_directories()
	return chromadb.PersistentClient(path=str(CHROMA_DIR))


def save_chunk_records(records: list[ChunkRecord]) -> None:
	ensure_directories()
	with CHUNKS_PATH.open("w", encoding="utf-8") as f:
		for record in records:
			f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def load_chunk_records() -> list[ChunkRecord]:
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
	return {r.chunk_id: r for r in load_chunk_records()}


def resolve_repo_path(path: str) -> Path:
	p = Path(path)
	if p.is_absolute():
		return p
	return Path(__file__).resolve().parent.parent / p

