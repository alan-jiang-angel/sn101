"""Miner candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CandidatePolicy:
    exclude_uid: int | None = None
    exclude_validator_permit: bool = False
    exclude_positive_validator_trust: bool = False


def _as_sequence(value: Any, n: int, default: Any) -> Sequence[Any]:
    if value is None:
        return [default] * n
    return value


def miner_candidates(metagraph: Any, policy: CandidatePolicy) -> list[int]:
    hotkeys = list(getattr(metagraph, "hotkeys", []))
    n = int(getattr(metagraph, "n", len(hotkeys)))
    validator_permit = _as_sequence(
        getattr(metagraph, "validator_permit", None), n, False
    )
    validator_trust = _as_sequence(
        getattr(metagraph, "validator_trust", None), n, 0.0
    )

    candidates: list[int] = []
    for uid in range(n):
        if policy.exclude_uid is not None and uid == policy.exclude_uid:
            continue
        if policy.exclude_validator_permit and bool(validator_permit[uid]):
            continue
        if (
            policy.exclude_positive_validator_trust
            and float(validator_trust[uid]) > 0
        ):
            continue
        candidates.append(uid)
    return candidates
