from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from agentic_rag.citations import extract_citation_strings


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized tool call requested by an AI message."""

    name: str
    args: dict[str, Any]
    id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A normalized tool result message."""

    name: str
    content: str
    parsed: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ToolCallStep:
    """A trace step containing one or more tool calls and their results."""

    thought: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]


@dataclass(frozen=True, slots=True)
class ThoughtStep:
    """An intermediate AI message without tool calls."""

    content: str


@dataclass(frozen=True, slots=True)
class FinalStep:
    """The final answer in an agent trace."""

    content: str
    citations: list[str]


TraceStep = ToolCallStep | ThoughtStep | FinalStep


def parse_agent_trace(result_raw: dict[str, Any]) -> list[TraceStep]:
    """Convert a LangGraph result dict into normalized trace steps."""
    return parse_messages(result_raw.get("messages", []))


def parse_messages(messages: list[Any]) -> list[TraceStep]:
    """Convert raw LangChain messages into normalized trace steps."""
    steps: list[TraceStep] = []
    index = 0
    while index < len(messages) and getattr(messages[index], "type", "") in ("system", "human"):
        index += 1

    while index < len(messages):
        message = messages[index]
        if getattr(message, "type", "") != "ai":
            index += 1
            continue

        tool_calls = _parse_tool_calls(message)
        if tool_calls:
            raw_results = []
            next_index = index + 1
            while (
                next_index < len(messages) and getattr(messages[next_index], "type", "") == "tool"
            ):
                raw_results.append(messages[next_index])
                next_index += 1
            steps.append(
                ToolCallStep(
                    thought=get_message_content(message),
                    tool_calls=tool_calls,
                    tool_results=[_parse_tool_result_message(result) for result in raw_results],
                )
            )
            index = next_index
            continue

        content = get_message_content(message)
        is_last_ai = not any(
            getattr(next_message, "type", "") == "ai" for next_message in messages[index + 1 :]
        )
        if is_last_ai:
            steps.append(FinalStep(content=content, citations=extract_citation_strings(content)))
        else:
            steps.append(ThoughtStep(content=content))
        index += 1

    return steps


def get_message_content(message: Any) -> str:
    """Return message content as plain text."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def parse_tool_result(content: str) -> dict[str, Any] | None:
    """Parse a serialized tool result dict, tolerating JSON and Python reprs."""
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(content)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def _parse_tool_result_message(message: Any) -> ToolResult:
    content = get_message_content(message)
    return ToolResult(
        name=str(getattr(message, "name", "tool")),
        content=content,
        parsed=parse_tool_result(content),
    )


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    raw = getattr(message, "tool_calls", None)
    if not raw:
        raw = getattr(message, "additional_kwargs", {}).get("tool_calls", [])
    for tool_call in raw:
        if not isinstance(tool_call, dict):
            continue

        name = str(tool_call.get("name", ""))
        args_raw = tool_call.get("args", {})
        id_ = str(tool_call.get("id", ""))
        if not name:
            function = tool_call.get("function", {})
            name = str(function.get("name", "unknown"))
            args_str = function.get("arguments", "{}")
            try:
                args_raw = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args_raw = {"raw": args_str}
        if isinstance(args_raw, str):
            try:
                args_raw = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args_raw = {"raw": args_raw}
        if not isinstance(args_raw, dict):
            args_raw = {"raw": args_raw}
        calls.append(ToolCall(name=name, args=args_raw, id=id_))
    return calls
