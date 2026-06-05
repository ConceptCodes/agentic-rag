from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


@dataclass(slots=True)
class AgentContext:
    """Per-invocation context passed to the agent graph."""

    user_id: str | None = None


class AgentGraphState(TypedDict, total=False):
    """State schema for the LangGraph agent execution graph."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    read_chunk_ids: list[str]
    tool_trace: list[dict[str, Any]]
    citations: list[str]


class RuntimeSettings(TypedDict):
    """Configuration settings available to the agent at runtime."""

    thread_id: str
    embedding_backend: str
    top_k: int
    max_steps: int
    model: str


class CitationItem(TypedDict):
    """A single citation entry linking a claim to a source chunk."""

    source: str
    chunk_id: str
    score: float | None
    snippet: str
