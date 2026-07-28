"""Drop-in replacement for tag101.tasks.sn101.

Registered via `--task.miner_module sn101_edge.task`. The miner's registry
registers override modules with replace=True, so exporting the same `kind`
swaps our solver in for the reference one. Because this package lives outside
the tag101 checkout, `git pull --ff-only` during auto-update cannot clobber it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tag101.tasks.framework import ScoreBreakdown, TaskHandler

from .tagger import solve_envelope

KIND = "sn101.tags.v1"
SPEC_VERSION = "v1"


def solve_problem(envelope: Any, chain_runtime: Any) -> dict[str, Any]:
    try:
        return solve_envelope(envelope)
    except Exception:  # noqa: BLE001
        # An empty answer scores zero but keeps the process alive; never let a
        # solver bug take the miner offline.
        return {"tags": []}


def score_answers(
    payload: Mapping[str, Any],
    scoring: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> ScoreBreakdown:
    """Delegate scoring to the reference implementation.

    Miners never score, so this is imported lazily to keep sklearn and
    sentence-transformers off the miner's startup path.
    """
    from tag101.tasks.sn101 import score_answers as reference_score

    return reference_score(payload, scoring, answers)


solve = solve_problem
score_batch = score_answers


def handler() -> TaskHandler:
    return TaskHandler(
        kind=KIND,
        spec_version=SPEC_VERSION,
        solve_problem=solve_problem,
        score_answers=score_answers,
        description="SN101 edge miner: crowd-simulating consensus optimiser.",
    )
