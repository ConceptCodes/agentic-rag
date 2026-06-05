from __future__ import annotations

import ast
import json
import re
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


def _get_content(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def _parse_tool_result(content: str) -> dict[str, Any] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(content)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def _parse_tool_calls(msg: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        raw = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args_raw = tc.get("args", {})
            id_ = tc.get("id", "")
            if not name:
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args_str = fn.get("arguments", "{}")
                try:
                    args_raw = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args_raw = {"raw": args_str}
            if isinstance(args_raw, str):
                try:
                    args_raw = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    args_raw = {"raw": args_raw}
            calls.append({"name": name, "args": args_raw, "id": id_})
    return calls


def _parse_messages(messages: list[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and getattr(messages[i], "type", "") in ("system", "human"):
        i += 1
    while i < len(messages):
        msg = messages[i]
        if getattr(msg, "type", "") == "ai":
            tool_calls = _parse_tool_calls(msg)
            if tool_calls:
                results = []
                j = i + 1
                while j < len(messages) and getattr(messages[j], "type", "") == "tool":
                    results.append(messages[j])
                    j += 1
                steps.append(
                    {
                        "type": "tool_call",
                        "thought": _get_content(msg),
                        "tool_calls": tool_calls,
                        "tool_results": results,
                    }
                )
                i = j
            else:
                content = _get_content(msg)
                is_last = not any(getattr(m, "type", "") == "ai" for m in messages[i + 1 :])
                steps.append({"type": "final" if is_last else "thought", "content": content})
                i += 1
        else:
            i += 1
    return steps


def _render_args(args: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in args.items():
        if isinstance(value, list):
            items = ", ".join(str(v)[:60] for v in value)
            lines.append(f"  [bold]{key}[/bold]  [{len(value)} items] {items}")
        elif isinstance(value, str) and len(value) > 80:
            lines.append(f"  [bold]{key}[/bold]  {value[:80]}...")
        else:
            lines.append(f"  [bold]{key}[/bold]  {value}")
    return "\n".join(lines)


def _render_tool_input(name: str, args: dict[str, Any]) -> Panel:
    return Panel(
        _render_args(args),
        title=f"[bold yellow]{name}[/bold yellow]",
        title_align="left",
        border_style="yellow",
        box=box.SIMPLE,
    )


def _render_keyword_table(results: list[dict[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Source", max_width=25)
    table.add_column("Snippet", max_width=80)
    for i, r in enumerate(results, 1):
        score = r.get("score")
        score_str = f"{score:.2f}" if score is not None else "—"
        table.add_row(
            str(i),
            score_str,
            str(r.get("source", "?"))[:25],
            str(r.get("snippet", ""))[:80],
        )
    return table


def _render_chunk_panels(results: list[dict[str, Any]], max_text: int = 400) -> list[Panel]:
    panels: list[Panel] = []
    for r in results:
        citation = r.get("citation", "")
        title = r.get("title", "")
        text = r.get("text", "")
        header = f"[bold]{citation}[/bold]"
        if title:
            header += f"  \u2014  {title[:60]}"
        truncated = text[:max_text] + ("..." if len(text) > max_text else "")
        panels.append(
            Panel(
                truncated,
                title=header,
                title_align="left",
                border_style="green",
                box=box.SIMPLE,
            )
        )
    return panels


def render_agent_run(
    question: str,
    result_raw: dict[str, Any],
    thread_id: str = "",
    model_name: str = "",
) -> None:
    """
    Render a detailed step-by-step trace of the agent's execution using Rich.

    Parses the raw LangGraph state messages into displayable steps (thoughts,
    tool calls, tool results) and prints them with styled panels, tables, and rules.
    """
    console = Console()
    messages = result_raw.get("messages", [])

    if not messages:
        console.print("[red]No agent output to show.[/red]")
        return

    steps = _parse_messages(messages)
    tool_steps = [s for s in steps if s["type"] == "tool_call"]
    total_calls = sum(len(s["tool_calls"]) for s in tool_steps)

    console.print()
    header = Group(
        Text("  Question   ", style="bold") + Text(question),
        Text("  Model      ", style="bold") + Text(model_name or "default"),
        Text("  Thread     ", style="bold") + Text(thread_id or "default"),
        Text("  Steps      ", style="bold")
        + Text(f"{len(tool_steps)} iterations \u00b7 {total_calls} tool calls"),
    )
    console.print(
        Panel(
            header,
            title="[bold blue]Agentic RAG[/bold blue] \u00b7 [bold]Showcase[/bold]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )
    console.print()

    step_num = 0
    for step in steps:
        if step["type"] == "tool_call":
            step_num += 1
            tc = len(step["tool_calls"])
            label = f"  Step {step_num} \u00b7 {tc} tool call{'s' if tc != 1 else ''}  "
            console.print(Rule(Text(label, style="bold yellow")))
            console.print()

            if step["thought"]:
                console.print(Text(step["thought"], style="cyan italic"))
                console.print()

            for tcall in step["tool_calls"]:
                console.print(_render_tool_input(tcall["name"], tcall["args"]))

            for tr in step["tool_results"]:
                content = _get_content(tr)
                parsed = _parse_tool_result(content)
                if parsed:
                    results = parsed.get("results", [])
                    tool_name = parsed.get("tool", getattr(tr, "name", "tool"))
                    if tool_name in ("keyword_search", "semantic_search"):
                        if results:
                            console.print(
                                Panel(
                                    _render_keyword_table(results),
                                    title=f"[bold green]Results ({len(results)})[/bold green]",
                                    title_align="left",
                                    border_style="green",
                                    box=box.SIMPLE,
                                )
                            )
                        else:
                            console.print(
                                Panel(
                                    "[dim]No results found[/dim]",
                                    border_style="green",
                                    box=box.SIMPLE,
                                )
                            )
                    elif tool_name == "chunk_read":
                        chunks = _render_chunk_panels(results)
                        for c in chunks:
                            console.print(c)
                    else:
                        text = content[:300] + ("..." if len(content) > 300 else "")
                        console.print(Panel(text, border_style="green", box=box.SIMPLE))
                else:
                    text = content[:300] + ("..." if len(content) > 300 else "")
                    console.print(Panel(text, border_style="green", box=box.SIMPLE))
                console.print()

        elif step["type"] == "thought":
            console.print(Rule(Text("  Intermediate thought  ", style="dim")))
            console.print()
            console.print(Text(step["content"], style="cyan italic"))
            console.print()

    final_steps = [s for s in steps if s["type"] == "final"]
    if final_steps:
        console.print()
        answer_text = final_steps[0]["content"]
        citations = sorted(set(re.findall(r"\[source:\s*[\w./\-]+/[\w.\-:]+\]", answer_text)))

        parts: list[Text] = [Text(answer_text)]
        if citations:
            parts.append(Text(""))
            parts.append(Text("Citations:", style="bold underline"))
            for c in citations:
                parts.append(Text(f"  \u2022 {c}"))

        console.print(
            Panel(
                Group(*parts),
                title="[bold green]Final Answer[/bold green]",
                border_style="green",
                box=box.HEAVY,
            )
        )
        console.print()
