"""Scoring helpers shared by task modules and validators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field


class MinerScore(BaseModel):
    """Score for one miner answer before it is packed into a validator report."""

    reward: float
    valid: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    """Batch scoring result returned by a task's ``score_answers`` hook."""

    rewards: list[float]
    valid: list[bool] = Field(default_factory=list)
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_miner_scores(
        cls,
        scores: Iterable[MinerScore],
        *,
        details: dict[str, Any] | None = None,
    ) -> "ScoreBreakdown":
        items = list(scores)
        return cls(
            rewards=[float(item.reward) for item in items],
            valid=[bool(item.valid) for item in items],
            metrics=[dict(item.metrics) for item in items],
            details=details or {},
        )

    def report_metadata(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"rewards"}, exclude_none=True)
        return {
            key: value
            for key, value in data.items()
            if value not in ({}, [])
        }
