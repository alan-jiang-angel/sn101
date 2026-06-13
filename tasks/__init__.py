"""Task leasing, solving, and scoring interfaces."""

from .catalog import build_task_registry
from .framework import (
    AnswerScorer,
    LeaseRequest,
    MinerAxonAnnouncement,
    MinerAxonRequest,
    MinerAxonResponse,
    MinerResult,
    MinerScore,
    MinerSolver,
    ResultReport,
    ScoreboardRequest,
    ScoreboardSnapshot,
    ScoreBreakdown,
    SignedLeaseRequest,
    SignedMinerAxonAnnouncement,
    SignedMinerAxonRequest,
    SignedResultReport,
    SignedScoreboardRequest,
    SignedScoreboardSnapshot,
    TaskAnswer,
    TaskHandler,
    TaskPayload,
    TaskProfile,
    TaskRegistry,
    TaskLease,
    default_registry,
)


def __getattr__(name: str):
    if name == "TaskServerClient":
        from .framework.client import TaskServerClient

        return TaskServerClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AnswerScorer",
    "LeaseRequest",
    "MinerAxonAnnouncement",
    "MinerAxonRequest",
    "MinerAxonResponse",
    "MinerResult",
    "MinerScore",
    "MinerSolver",
    "ResultReport",
    "ScoreboardRequest",
    "ScoreboardSnapshot",
    "ScoreBreakdown",
    "SignedLeaseRequest",
    "SignedMinerAxonAnnouncement",
    "SignedMinerAxonRequest",
    "SignedResultReport",
    "SignedScoreboardRequest",
    "SignedScoreboardSnapshot",
    "TaskAnswer",
    "TaskHandler",
    "TaskPayload",
    "TaskProfile",
    "TaskRegistry",
    "TaskServerClient",
    "TaskLease",
    "build_task_registry",
    "default_registry",
]
