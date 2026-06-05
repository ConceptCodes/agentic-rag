from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from src.constants import DEFAULT_CHAT_MODEL, DEFAULT_MAX_AGENT_STEPS, LMSTUDIO_BASE_URL
from src.prompts import build_system_prompt
from src.state import AgentContext, AgentGraphState
from src.tools import ALL_TOOLS


@dataclass(slots=True)
class AgentResult:
	answer: str
	citations: list[str]
	raw: dict[str, Any]


def _build_model(model_name: str, base_url: str = LMSTUDIO_BASE_URL) -> ChatOpenAI:
	api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
	return ChatOpenAI(
		model=model_name,
		base_url=base_url,
		api_key=api_key,
		temperature=0,
	)


def build_agent(model_name: str = DEFAULT_CHAT_MODEL):
	model = _build_model(model_name=model_name)
	return create_agent(
		model=model,
		tools=ALL_TOOLS,
		system_prompt=build_system_prompt(),
		context_schema=AgentContext,
		state_schema=AgentGraphState,
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

	citations = sorted(set(re.findall(r"\[source:\s*[^\]]+\]", answer)))
	return AgentResult(answer=answer.strip(), citations=citations, raw=result)

