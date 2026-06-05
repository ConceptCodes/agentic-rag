from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPANY_DOCS_DIR = DATA_DIR / "company_docs"
EVAL_DATA_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBEDDING_BACKEND = "sentence_transformers"
DEFAULT_EMBEDDING_MODEL_SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_LMSTUDIO = "gemmaembedding-370m"

# Per A-RAG, chunks are up to 1000 tokens while preserving sentence boundaries.
DEFAULT_CHUNK_MAX_TOKENS = 1000
DEFAULT_TOP_K = 5
DEFAULT_MAX_AGENT_STEPS = 12

DEFAULT_EVAL_SAMPLES = 10
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
HOTPOT_DATASET_SPLIT = "validation"

ANSWER_CITATION_PATTERN = "[source: doc_id/chunk_id]"

