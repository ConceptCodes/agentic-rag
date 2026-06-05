from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.citations import extract_citation_strings
from agentic_rag.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_OPENROUTER_CHAT_MODEL,
    LMSTUDIO_BASE_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_DEEPSEEK_V4_PRO_MODEL,
    OPENROUTER_GPT_5_MINI_MODEL,
)
from agentic_rag.prompts import SYSTEM_PROMPT
from agentic_rag.state import AgentContext, AgentGraphState
from agentic_rag.tools import ALL_TOOLS, reset_context_tracker


@dataclass(slots=True)
class AgentResult:
    """Result of an agent invocation including answer and extracted citations."""

    answer: str
    citations: list[str]
    raw: dict[str, Any]


MODEL_ALIASES = {
    "deepseek-v4-pro": OPENROUTER_DEEPSEEK_V4_PRO_MODEL,
    "gpt-5-mini": OPENROUTER_GPT_5_MINI_MODEL,
    "lmstudio": DEFAULT_CHAT_MODEL,
}


def resolve_model_name(
    model_name: str | None,
    chat_provider: str = DEFAULT_CHAT_PROVIDER,
) -> str:
    """Resolve default and short alias model names."""
    if model_name:
        return MODEL_ALIASES.get(model_name, model_name)
    if chat_provider == "openrouter":
        return DEFAULT_OPENROUTER_CHAT_MODEL
    return DEFAULT_CHAT_MODEL


def _openrouter_headers() -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = title
    return headers or None


def _reasoning_config(
    reasoning_enabled: bool,
    reasoning_effort: str | None = None,
) -> dict[str, object] | None:
    if not reasoning_enabled:
        return None
    config: dict[str, object] = {"enabled": True}
    if reasoning_effort:
        config["effort"] = reasoning_effort
    return config


def _build_model(
    model_name: str,
    chat_provider: str = DEFAULT_CHAT_PROVIDER,
    reasoning_enabled: bool = True,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    if chat_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            msg = "OPENROUTER_API_KEY must be set when using --chat-provider openrouter"
            raise ValueError(msg)
        return ChatOpenAI(
            model=model_name,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0,
            default_headers=_openrouter_headers(),
            reasoning=_reasoning_config(reasoning_enabled, reasoning_effort),
            model_kwargs={"parallel_tool_calls": False},
        )

    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    return ChatOpenAI(
        model=model_name,
        base_url=LMSTUDIO_BASE_URL,
        api_key=api_key,  # type: ignore[arg-type]
        temperature=0,
        model_kwargs={"parallel_tool_calls": False},
    )


@functools.lru_cache(maxsize=4)
def build_agent(
    model_name: str = DEFAULT_CHAT_MODEL,
    chat_provider: str = DEFAULT_CHAT_PROVIDER,
    reasoning_enabled: bool = True,
    reasoning_effort: str | None = None,
) -> Any:
    """Build a cached LangGraph agent with the given model."""
    model = _build_model(
        model_name=model_name,
        chat_provider=chat_provider,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
    )
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
    model_name: str | None = None,
    chat_provider: str = DEFAULT_CHAT_PROVIDER,
    reasoning_enabled: bool = True,
    reasoning_effort: str | None = None,
    max_steps: int = DEFAULT_MAX_AGENT_STEPS,
    user_id: str | None = None,
) -> AgentResult:
    """Invoke the agent with a question and return the answer with citations."""
    reset_context_tracker()
    resolved_model_name = resolve_model_name(model_name, chat_provider=chat_provider)
    agent = build_agent(
        model_name=resolved_model_name,
        chat_provider=chat_provider,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "recursion_limit": max_steps,
            "configurable": {
                "thread_id": thread_id,
                "chat_provider": chat_provider,
                "model": resolved_model_name,
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

    citations = extract_citation_strings(answer)
    return AgentResult(answer=answer.strip(), citations=citations, raw=result)
