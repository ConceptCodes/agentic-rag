from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agentic_rag.trace import FinalStep, ThoughtStep, ToolCallStep, parse_agent_trace


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
    steps = parse_agent_trace(result_raw)

    if not steps:
        console.print("[red]No agent output to show.[/red]")
        return

    tool_steps = [step for step in steps if isinstance(step, ToolCallStep)]
    total_calls = sum(len(step.tool_calls) for step in tool_steps)

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
        if isinstance(step, ToolCallStep):
            step_num += 1
            tc = len(step.tool_calls)
            label = f"  Step {step_num} \u00b7 {tc} tool call{'s' if tc != 1 else ''}  "
            console.print(Rule(Text(label, style="bold yellow")))
            console.print()

            if step.thought:
                console.print(Text(step.thought, style="cyan italic"))
                console.print()

            for tool_call in step.tool_calls:
                console.print(_render_tool_input(tool_call.name, tool_call.args))

            for tool_result in step.tool_results:
                if tool_result.parsed:
                    parsed = tool_result.parsed
                    results = parsed.get("results", [])
                    tool_name = parsed.get("tool", tool_result.name)
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
                        for chunk_panel in chunks:
                            console.print(chunk_panel)
                    else:
                        text = tool_result.content[:300] + (
                            "..." if len(tool_result.content) > 300 else ""
                        )
                        console.print(Panel(text, border_style="green", box=box.SIMPLE))
                else:
                    text = tool_result.content[:300] + (
                        "..." if len(tool_result.content) > 300 else ""
                    )
                    console.print(Panel(text, border_style="green", box=box.SIMPLE))
                console.print()

        elif isinstance(step, ThoughtStep):
            console.print(Rule(Text("  Intermediate thought  ", style="dim")))
            console.print()
            console.print(Text(step.content, style="cyan italic"))
            console.print()

    final_steps = [step for step in steps if isinstance(step, FinalStep)]
    if final_steps:
        console.print()
        answer_text = final_steps[0].content
        citations = final_steps[0].citations

        parts: list[Text] = [Text(answer_text)]
        if citations:
            parts.append(Text(""))
            parts.append(Text("Citations:", style="bold underline"))
            for citation in citations:
                parts.append(Text(f"  \u2022 {citation}"))

        console.print(
            Panel(
                Group(*parts),
                title="[bold green]Final Answer[/bold green]",
                border_style="green",
                box=box.HEAVY,
            )
        )
        console.print()
