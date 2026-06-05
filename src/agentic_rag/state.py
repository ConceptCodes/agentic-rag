from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


@dataclass(slots=True)
class AgentContext:
    """Per-invocation context passed to the agent graph."""

    user_id: str | None = None


class AgentGraphState(TypedDict, total=False):
    """State schema for the LangGraph agent execution graph."""

    messages: Annotated[list[AnyMessage], operator.add]
