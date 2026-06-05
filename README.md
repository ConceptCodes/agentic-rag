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

- [main.py](https://github.com/ConceptCodes/agentic-rag/blob/main/main.py): CLI (`ingest`, `ask`, `eval`)
- [src/agentic_rag/constants.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/constants.py): defaults and paths
- [src/agentic_rag/state.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/state.py): state/context contracts
- [src/agentic_rag/ingest.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/ingest.py): markdown ingestion
- [src/agentic_rag/corpus.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/corpus.py): indexed corpus storage and retrieval
- [src/agentic_rag/embeddings.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/embeddings.py): embedding backend adapters
- [src/agentic_rag/tools.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/tools.py): LangChain retrieval tool wrappers
- [src/agentic_rag/agent.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/agent.py): agent harness creation and invocation
- [src/agentic_rag/prompts.py](https://github.com/ConceptCodes/agentic-rag/blob/main/src/agentic_rag/prompts.py): system prompt + citation policy

## Setup

```bash
uv sync
```

If using LM Studio, ensure:

- Server is running at `http://localhost:1234/v1`
- A chat model is loaded for `ask`
- Optionally, an embedding model is loaded for `--embedding-backend lmstudio`

If using OpenRouter, set:

```bash
export OPENROUTER_API_KEY="..."
```

Optional OpenRouter leaderboard headers:

```bash
export OPENROUTER_HTTP_REFERER="https://your-site.example"
export OPENROUTER_APP_TITLE="Agentic RAG"
```

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

OpenRouter model examples:

```bash
uv run agentic-rag ask "How does Convex handle mutations?" \
  --chat-provider openrouter \
  --model deepseek-v4-pro \
  --reasoning-effort xhigh \
  --showcase

uv run agentic-rag ask "When should I use Cloudflare D1 instead of KV?" \
  --chat-provider openrouter \
  --model gpt-5-mini \
  --reasoning-effort high \
  --showcase
```

The agent uses a ReAct loop, so the OpenRouter examples use reasoning-capable
models by default. Reasoning is enabled unless you pass `--no-reasoning`.

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
