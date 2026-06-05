from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

from langchain.messages import AnyMessage
from typing_extensions import Annotated, TypedDict


@dataclass(slots=True)
class AgentContext:
	user_id: str | None = None


class AgentGraphState(TypedDict, total=False):
	messages: Annotated[list[AnyMessage], operator.add]
	llm_calls: int
	read_chunk_ids: list[str]
	tool_trace: list[dict[str, Any]]
	citations: list[str]


class RuntimeSettings(TypedDict):
	thread_id: str
	embedding_backend: str
	top_k: int
	max_steps: int
	model: str


class CitationItem(TypedDict):
	source: str
	chunk_id: str
	score: float | None
	snippet: str

