from __future__ import annotations

import argparse
import json
import logging
import sys

from agentic_rag.agent import invoke_agent
from agentic_rag.constants import DEFAULT_CHAT_MODEL, DEFAULT_EVAL_SAMPLES
from agentic_rag.ingest import index_documents
from eval.datasets import fetch_hotpotqa_subset

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
        embedding_backend=args.embedding_backend,
        top_k=args.top_k,
        max_steps=args.max_steps,
        user_id=args.user_id,
    )
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
    samples = fetch_hotpotqa_subset(limit=args.samples)
    score = 0
    for i, item in enumerate(samples, start=1):
        question = item.get("question", "")
        expected = str(item.get("answer", "")).strip().lower()
        try:
            result = invoke_agent(
                question=question,
                thread_id=f"eval-{i}",
                model_name=args.model,
                embedding_backend=args.embedding_backend,
                top_k=args.top_k,
                max_steps=args.max_steps,
            )
            got = result.answer.lower()
            hit = bool(expected and expected in got)
            score += int(hit)
        except Exception:
            logger.exception("Evaluation sample %d failed", i)
            hit = False
        print(f"[{i}/{len(samples)}] hit={hit} question={question[:80]!r}")

    summary: dict[str, object] = {
        "samples": len(samples),
        "contain_match_accuracy": (score / len(samples)) if samples else 0,
        "hits": score,
    }
    print("\n" + json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest markdown docs into chunk index")
    ingest_p.add_argument(
        "--embedding-backend",
        default="sentence_transformers",
        choices=["sentence_transformers", "lmstudio"],
    )
    ingest_p.set_defaults(func=cmd_ingest)

    ask_p = sub.add_parser("ask", help="Ask the retrieval agent a question")
    ask_p.add_argument("question")
    ask_p.add_argument("--thread-id", default="local-thread")
    ask_p.add_argument("--user-id", default=None)
    ask_p.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    ask_p.add_argument(
        "--embedding-backend",
        default="sentence_transformers",
        choices=["sentence_transformers", "lmstudio"],
    )
    ask_p.add_argument("--top-k", type=int, default=5)
    ask_p.add_argument("--max-steps", type=int, default=12)
    ask_p.set_defaults(func=cmd_ask)

    eval_p = sub.add_parser("eval", help="Run lightweight HotpotQA subset evaluation")
    eval_p.add_argument("--samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    eval_p.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    eval_p.add_argument(
        "--embedding-backend",
        default="sentence_transformers",
        choices=["sentence_transformers", "lmstudio"],
    )
    eval_p.add_argument("--top-k", type=int, default=5)
    eval_p.add_argument("--max-steps", type=int, default=12)
    eval_p.set_defaults(func=cmd_eval)

    return parser


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
