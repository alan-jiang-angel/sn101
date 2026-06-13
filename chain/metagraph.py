"""Small metagraph eligibility helpers shared by validator and server code."""

from __future__ import annotations

from typing import Any

SUBNET_OWNER_UID = 0
DEFAULT_SCOREBOARD_MAX_AGE_BLOCKS = 1000


def block_within_window(
    *,
    current_block: int,
    signed_block: int,
    max_age_blocks: int = DEFAULT_SCOREBOARD_MAX_AGE_BLOCKS,
) -> bool:
    try:
        current = int(current_block)
        signed = int(signed_block)
    except (TypeError, ValueError):
        return False
    if current < 0 or signed < 0:
        return False
    return abs(current - signed) <= int(max_age_blocks)


def non_owner_has_incentive(
    metagraph: Any,
    uid: int,
    *,
    owner_uid: int = SUBNET_OWNER_UID,
) -> bool:
    """Return True when a non-owner UID has positive incentive.

    Localnet setup treats UID 0 as the subnet owner. The owner is allowed to
    have incentive, but other score-signing validators are not.
    """

    uid = int(uid)
    if uid == int(owner_uid):
        return False
    return incentive_at_uid(metagraph, uid) > 0.0


def incentive_at_uid(metagraph: Any, uid: int) -> float:
    incentives = _incentive_values(metagraph)
    uid = int(uid)
    if uid < 0 or uid >= len(incentives):
        return 0.0
    return _numeric_as_float(incentives[uid])


def _incentive_values(metagraph: Any) -> list[Any]:
    for name in ("I", "incentive", "incentives"):
        if hasattr(metagraph, name):
            value = getattr(metagraph, name)
            break
    else:
        return []

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _numeric_as_float(value: Any) -> float:
    tao = getattr(value, "tao", None)
    if tao is not None:
        return float(tao)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
