from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("AGENTIC_RAG_ROOT", Path.cwd())).resolve()
DATA_DIR = PROJECT_ROOT / "data"
COMPANY_DOCS_DIR = DATA_DIR / "company_docs"
EVAL_DATA_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_PROVIDER = "lmstudio"
DEFAULT_LMSTUDIO_CHAT_MODEL = "google/gemma-4-e4b"
OPENROUTER_DEEPSEEK_V4_PRO_MODEL = "deepseek/deepseek-v4-pro"
OPENROUTER_GPT_5_MINI_MODEL = "openai/gpt-5-mini"
DEFAULT_OPENROUTER_CHAT_MODEL = OPENROUTER_DEEPSEEK_V4_PRO_MODEL
DEFAULT_CHAT_MODEL = DEFAULT_LMSTUDIO_CHAT_MODEL
DEFAULT_EMBEDDING_BACKEND = "sentence_transformers"
DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_LMSTUDIO = "gemmaembedding-370m"

# Per A-RAG, chunks are up to 1000 tokens while preserving sentence boundaries.
DEFAULT_CHUNK_MAX_TOKENS = 1000
DEFAULT_TOP_K = 5
DEFAULT_MAX_AGENT_STEPS = 12
