from __future__ import annotations

import argparse
import json
import logging
import sys

from agentic_rag.agent import invoke_agent
from agentic_rag.constants import (
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_MAX_AGENT_STEPS,
)
from agentic_rag.ingest import index_documents
from agentic_rag.showcase import render_agent_run
from eval.datasets import fetch_hotpotqa_subset
from eval.harness import DEFAULT_EVAL_SAMPLES, parse_eval_sample, run_eval

logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest markdown documents into the chunk index."""
    stats = index_documents(embedding_backend=args.embedding_backend)
    print(json.dumps({"status": "ok", **stats}, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask the retrieval agent a question and print the answer."""
    result = invoke_agent(
        question=args.question,
        thread_id=args.thread_id,
        model_name=args.model,
        chat_provider=args.chat_provider,
        reasoning_enabled=args.reasoning,
        reasoning_effort=args.reasoning_effort,
        max_steps=args.max_steps,
        user_id=args.user_id,
    )
    if args.showcase:
        render_agent_run(
            question=args.question,
            result_raw=result.raw,
            thread_id=args.thread_id,
            model_name=args.model or "default",
        )
    else:
        print("Answer:\n")
        print(result.answer)
        print("\nCitations:")
        if not result.citations:
            print("- none")
        else:
            for c in result.citations:
                print(f"- {c}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run a contain-match evaluation against a HotpotQA subset."""
    rows = fetch_hotpotqa_subset(limit=args.samples)
    samples = [parse_eval_sample(row) for row in rows]

    def ask_agent(question: str, thread_id: str) -> str:
        return invoke_agent(
            question=question,
            thread_id=thread_id,
            model_name=args.model,
            chat_provider=args.chat_provider,
            reasoning_enabled=args.reasoning,
            reasoning_effort=args.reasoning_effort,
            max_steps=args.max_steps,
        ).answer

    report = run_eval(samples, ask_agent, logger=logger)
    for result in report.results:
        print(
            f"[{result.index}/{report.samples}] hit={result.hit} "
            f"question={result.sample.question[:80]!r}"
        )

    print("\n" + json.dumps(report.summary(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest markdown docs into chunk index")
    ingest_p.add_argument(
        "--embedding-backend",
        default=DEFAULT_EMBEDDING_BACKEND,
        choices=["sentence_transformers", "lmstudio"],
    )
    ingest_p.set_defaults(func=cmd_ingest)

    ask_p = sub.add_parser("ask", help="Ask the retrieval agent a question")
    ask_p.add_argument("question")
    ask_p.add_argument("--thread-id", default="local-thread")
    ask_p.add_argument("--user-id", default=None)
    _add_chat_model_args(ask_p)
    ask_p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_AGENT_STEPS)
    ask_p.add_argument(
        "--showcase",
        action="store_true",
        help="Show a detailed step-by-step trace of the agent's execution",
    )
    ask_p.set_defaults(func=cmd_ask)

    eval_p = sub.add_parser("eval", help="Run lightweight HotpotQA subset evaluation")
    eval_p.add_argument("--samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    _add_chat_model_args(eval_p)
    eval_p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_AGENT_STEPS)
    eval_p.set_defaults(func=cmd_eval)

    return parser


def _add_chat_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chat-provider",
        default=DEFAULT_CHAT_PROVIDER,
        choices=["lmstudio", "openrouter"],
        help="Chat model provider to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name or alias. Aliases: deepseek-v4-pro, gpt-5-mini, lmstudio. "
            "Defaults depend on --chat-provider."
        ),
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable OpenRouter reasoning for models that support it",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Optional OpenRouter reasoning effort",
    )


def main() -> None:
    """Entry point for the Agentic RAG CLI."""
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
