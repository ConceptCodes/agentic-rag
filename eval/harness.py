from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_EVAL_SAMPLES = 10


@dataclass(frozen=True, slots=True)
class EvalSample:
    """A single question-answer evaluation sample."""

    question: str
    answer: str
    id: str = ""


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result for one evaluated sample."""

    index: int
    sample: EvalSample
    answer: str
    hit: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate contain-match evaluation report."""

    results: list[EvalResult]

    @property
    def samples(self) -> int:
        """Return the number of evaluated samples."""
        return len(self.results)

    @property
    def hits(self) -> int:
        """Return the number of samples that matched."""
        return sum(int(result.hit) for result in self.results)

    @property
    def contain_match_accuracy(self) -> float:
        """Return contain-match accuracy over evaluated samples."""
        return (self.hits / self.samples) if self.samples else 0.0

    def summary(self) -> dict[str, object]:
        """Return the JSON-serializable summary used by the CLI."""
        return {
            "samples": self.samples,
            "contain_match_accuracy": self.contain_match_accuracy,
            "hits": self.hits,
        }


AgentFn = Callable[[str, str], str]


def parse_eval_sample(row: dict[str, Any]) -> EvalSample:
    """Convert a dataset row into the evaluation sample shape."""
    return EvalSample(
        id=str(row.get("id", "")),
        question=str(row.get("question", "")),
        answer=str(row.get("answer", "")),
    )


def score_answer(expected: str, got: str) -> bool:
    """Score a response with the current contain-match metric."""
    expected_normalized = expected.strip().lower()
    return bool(expected_normalized and expected_normalized in got.lower())


def run_eval(
    samples: list[EvalSample],
    agent_fn: AgentFn,
    logger: logging.Logger | None = None,
) -> EvalReport:
    """Run evaluation samples through an agent function."""
    results: list[EvalResult] = []
    for index, sample in enumerate(samples, start=1):
        try:
            answer = agent_fn(sample.question, f"eval-{index}")
            results.append(
                EvalResult(
                    index=index,
                    sample=sample,
                    answer=answer,
                    hit=score_answer(sample.answer, answer),
                )
            )
        except Exception as exc:
            if logger:
                logger.exception("Evaluation sample %d failed", index)
            results.append(
                EvalResult(
                    index=index,
                    sample=sample,
                    answer="",
                    hit=False,
                    error=str(exc),
                )
            )
    return EvalReport(results=results)


def result_to_dict(result: EvalResult) -> dict[str, object]:
    """Return a JSON-serializable result row for future reporters."""
    return asdict(result)
