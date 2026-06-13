"""Generic task framework used by miners and validators.

Task authors provide one handler per task kind. Public handlers solve payloads
for miners and score miner answers for validators; task generation lives in
the task server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ...chain.runtime import ChainRuntime
from ...protocol import TaskEnvelope
from .scoring import ScoreBreakdown


TaskPayload = Mapping[str, Any]
TaskAnswer = Mapping[str, Any]
TaskProfile = Mapping[str, Any]

MinerSolver = Callable[[TaskEnvelope, ChainRuntime], dict[str, Any]]
AnswerScorer = Callable[[TaskPayload, Mapping[str, Any], Sequence[TaskAnswer]], ScoreBreakdown]

# Name aliases for task modules that still use the older hook labels.
SolveFn = MinerSolver
ScoreFn = AnswerScorer


@dataclass(frozen=True, init=False)
class TaskHandler:
    """Public miner/validator contract for one task kind.

    New task modules should provide two project-specific hooks:

    - ``solve_problem``: turns a task envelope and chain runtime into an answer.
    - ``score_answers``: scores miner answers using the scoring context.
    """

    kind: str
    spec_version: str
    solve_problem: MinerSolver
    score_answers: AnswerScorer
    description: str = ""

    def __init__(
        self,
        *,
        kind: str,
        spec_version: str,
        solve_problem: MinerSolver | None = None,
        score_answers: AnswerScorer | None = None,
        solve: MinerSolver | None = None,
        score_batch: AnswerScorer | None = None,
        description: str = "",
    ):
        solver = solve_problem or solve
        scorer = score_answers or score_batch
        if solver is None:
            raise ValueError("TaskHandler requires solve_problem")
        if scorer is None:
            raise ValueError("TaskHandler requires score_answers")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "spec_version", spec_version)
        object.__setattr__(self, "solve_problem", solver)
        object.__setattr__(self, "score_answers", scorer)
        object.__setattr__(self, "description", description)

    @property
    def solve(self) -> MinerSolver:
        return self.solve_problem

    @property
    def score_batch(self) -> AnswerScorer:
        return self.score_answers
