# Agentic RAG (A-RAG Style)

This project implements a local, tool-using retrieval agent inspired by:

Du et al., "A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces" (arXiv:2602.03442v1).

## Current Architecture

- Runtime: LangChain `create_agent` harness with LangGraph state/checkpoint semantics.
- Model: OpenAI-compatible chat endpoint (LM Studio recommended) through `langchain-openai`.
- Retrieval interfaces (paper-aligned):
	- `keyword_search`: simple keyword-based retrieval using Chroma's query capabilities.
	- `semantic_search`: dense retrieval using Chroma vector search with sentence-transformer embeddings.
	- `chunk_read`: direct chunk retrieval by ID from Chroma.
- Vector index: local Chroma persistent store.
- Embeddings:
	- `sentence_transformers` backend
	- `lmstudio` embeddings backend (OpenAI-compatible endpoint)

## Citation Contract

Agent answers must include inline citations for material claims using:

`[source: doc_id/chunk_id]`

Each citation should map to evidence returned by retrieval tools.

## Project Layout

- [main.py](main.py): CLI (`ingest`, `ask`, `eval`)
- [src/constants.py](src/constants.py): defaults and paths
- [src/state.py](src/state.py): state/context contracts
- [src/ingest.py](src/ingest.py): markdown ingestion and indexing
- [src/tools.py](src/tools.py): retrieval tools
- [src/agent.py](src/agent.py): agent harness creation and invocation
- [src/prompts.py](src/prompts.py): system prompt + citation policy
- [src/utils.py](src/utils.py): embeddings, Chroma, dataset helpers

## Setup

```bash
uv sync
```

If using LM Studio, ensure:

- Server is running at `http://localhost:1234/v1`
- A chat model is loaded for `ask`
- Optionally, an embedding model is loaded for `--embedding-backend lmstudio`

## Demo Docs

Put company markdown files under `data/company_docs/`.

Evaluation dataset snapshots are cached under `data/eval/`.

Chunking uses sentence-preserving packing with a hard cap of 1000 tokens per chunk.

Then index:

```bash
uv run agentic-rag ingest --embedding-backend sentence_transformers
```

## Ask

```bash
uv run agentic-rag ask "What is our onboarding policy?" --thread-id demo-1
```

## Eval (HotpotQA subset)

```bash
uv run agentic-rag eval --samples 10
```

This runs a lightweight contain-match evaluation against a sampled HotpotQA subset.

## References
- arXiv: https://arxiv.org/abs/2602.03442v1
- LangChain agents: https://docs.langchain.com/oss/python/langchain/agents
- LangChain tools: https://docs.langchain.com/oss/python/langchain/tools
- RunnableConfig reference: https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig
- ChromaDB: https://www.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- LM Studio: https://lmstudio.ai/
