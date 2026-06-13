"""Small non-graph task example for the generic task framework."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..chain.runtime import ChainRuntime
from ..protocol import TaskEnvelope
from .framework import (
    MinerScore,
    ScoreBreakdown,
    TaskHandler,
)


KIND = "arithmetic.v1"
SPEC_VERSION = "v1"


def solve_problem(envelope: TaskEnvelope, chain_runtime: ChainRuntime) -> dict[str, Any]:
    payload = envelope.payload
    left = int(payload["left"])
    right = int(payload["right"])
    operator = str(payload.get("operator", "sum"))
    value = left + right if operator == "sum" else left * right
    return {"value": value}


def _answer_value(answer: Mapping[str, Any]) -> int | None:
    try:
        return int(answer.get("value"))
    except (TypeError, ValueError):
        return None


def score_answers(
    payload: Mapping[str, Any],
    scoring: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> ScoreBreakdown:
    expected = int(scoring["expected"])
    scores: list[MinerScore] = []
    for answer in answers:
        value = _answer_value(answer)
        ok = value == expected
        scores.append(
            MinerScore(
                reward=1.0 if ok else 0.0,
                valid=ok,
                metrics={"value": value, "expected": expected},
            )
        )

    return ScoreBreakdown.from_miner_scores(
        scores,
        details={"method": scoring.get("method", "exact-match")},
    )


solve = solve_problem
score_batch = score_answers


def handler() -> TaskHandler:
    return TaskHandler(
        kind=KIND,
        spec_version=SPEC_VERSION,
        solve_problem=solve_problem,
        score_answers=score_answers,
        description="Arithmetic exact-match example task.",
    )
