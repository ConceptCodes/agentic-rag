from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.constants import DEFAULT_CHAT_MODEL, DEFAULT_MAX_AGENT_STEPS, LMSTUDIO_BASE_URL
from agentic_rag.prompts import SYSTEM_PROMPT
from agentic_rag.state import AgentContext, AgentGraphState
from agentic_rag.tools import ALL_TOOLS


@dataclass(slots=True)
class AgentResult:
    """Result of an agent invocation including answer and extracted citations."""

    answer: str
    citations: list[str]
    raw: dict[str, Any]


def _build_model(model_name: str, base_url: str = LMSTUDIO_BASE_URL) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,  # type: ignore[arg-type]
        temperature=0,
    )


@functools.lru_cache(maxsize=4)
def build_agent(model_name: str = DEFAULT_CHAT_MODEL) -> Any:
    """Build a cached LangGraph agent with the given model."""
    model = _build_model(model_name=model_name)
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        state_schema=AgentGraphState,  # type: ignore[arg-type]
        # TODO: migrate to langgraph.checkpoint.sqlite.SqliteSaver for
        # cross-process thread persistence.
        checkpointer=InMemorySaver(),
        name="agentic_rag",
    )


def invoke_agent(
    question: str,
    thread_id: str = "local-thread",
    model_name: str = DEFAULT_CHAT_MODEL,
    embedding_backend: str = "sentence_transformers",
    top_k: int = 5,
    max_steps: int = DEFAULT_MAX_AGENT_STEPS,
    user_id: str | None = None,
) -> AgentResult:
    """Invoke the agent with a question and return the answer with citations."""
    agent = build_agent(model_name=model_name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "recursion_limit": max_steps,
            "configurable": {
                "thread_id": thread_id,
                "model": model_name,
                "embedding_backend": embedding_backend,
                "top_k": top_k,
                "max_steps": max_steps,
            },
        },
        context=AgentContext(user_id=user_id),
    )
    messages = result.get("messages", [])
    answer = ""
    if messages:
        last = messages[-1]
        answer = getattr(last, "text", None) or getattr(last, "content", "") or ""
        if isinstance(answer, list):
            answer = " ".join(str(x) for x in answer)

    citations = sorted(set(re.findall(r"\[source:\s*[\w./\-]+/[\w.\-:]+\]", answer)))
    return AgentResult(answer=answer.strip(), citations=citations, raw=result)
