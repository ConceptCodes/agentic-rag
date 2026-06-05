# Agent Commands

## Setup
```bash
uv sync --group dev
uv pip install .
```

## Lint & Format
```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ruff format .
```

## Type Check
```bash
uv run --no-sync mypy src/agentic_rag eval
```

## Test
```bash
uv run --no-sync pytest
```

## Run
```bash
uv run --no-sync agentic-rag --help
uv run --no-sync agentic-rag ingest
uv run --no-sync agentic-rag ask "What is our onboarding policy?" --thread-id demo-1
uv run --no-sync agentic-rag eval --samples 10
```
