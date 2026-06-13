"""Validator reward state.

Subnet developers: this is the main place to customize how short-term task
rewards become long-term miner scores. Add a ``ScoreStrategy`` below, register
it in ``SCORE_STRATEGIES``, then run the validator with
``--validator.score_strategy <name>``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np


ScoreUpdateFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class ScoreReadContext:
    score_state: np.ndarray
    counts: np.ndarray
    alpha: float
    raw_history: list[dict[str, Any]]
    now: float
    window_seconds: float
    softmax_beta: float


ScoreReadFn = Callable[[ScoreReadContext], np.ndarray]


@dataclass(frozen=True)
class ScoreStrategy:
    """How a validator turns per-task rewards into long-term miner scores."""

    name: str
    update: ScoreUpdateFn
    read: ScoreReadFn
    description: str
    uses_raw_history: bool = False


def _ema_update(
    current: np.ndarray,
    rewards: np.ndarray,
    _counts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return alpha * rewards + (1.0 - alpha) * current


def _debiased_ema_scores(context: ScoreReadContext) -> np.ndarray:
    state = context.score_state
    counts = context.counts
    alpha = context.alpha
    correction = 1.0 - np.power(1.0 - alpha, counts)
    return np.divide(state, correction, out=np.zeros_like(state), where=correction > 0)


def _latest_update(
    _current: np.ndarray,
    rewards: np.ndarray,
    _counts: np.ndarray,
    _alpha: float,
) -> np.ndarray:
    return rewards


def _raw_scores(context: ScoreReadContext) -> np.ndarray:
    return np.asarray(context.score_state, dtype=np.float64)


def _softmax_observed_scores(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    if not np.any(observed):
        return output
    clean = np.nan_to_num(
        values[observed],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    shifted = beta * (clean - np.max(clean))
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return output
    output[observed] = exp_values / total
    return output


def _rolling_ma_softmax_scores(context: ScoreReadContext) -> np.ndarray:
    size = len(context.score_state)
    sums = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.int64)
    cutoff = float(context.now) - context.window_seconds
    for entry in context.raw_history:
        if float(entry.get("timestamp", 0.0)) < cutoff:
            continue
        uids = entry.get("uids", [])
        scores = entry.get("raw_scores", entry.get("scores", []))
        if not isinstance(uids, list) or not isinstance(scores, list):
            continue
        for uid_value, score_value in zip(uids, scores):
            uid = int(uid_value)
            if 0 <= uid < size:
                sums[uid] += float(score_value)
                counts[uid] += 1
    max_count = int(np.max(counts)) if np.any(counts > 0) else 0
    if max_count == 0:
        return np.zeros(size, dtype=np.float64)
    # Miners with fewer observations are averaged as if missing entries
    # were zero-valued scores.
    averages = sums / float(max_count)
    return _softmax_observed_scores(
        averages,
        counts > 0,
        beta=context.softmax_beta,
    )


# ---------------------------------------------------------------------------
# Subnet developer customization point
# ---------------------------------------------------------------------------
# Default behavior is a debiased EMA, which smooths noisy task rewards before
# weight submission. To try another policy:
#
# 1. Add update/read functions above.
# 2. Register a new ScoreStrategy below.
# 3. Run the validator with --validator.score_strategy <your_name>.
#
# Keep the strategy output as one float score per UID. The weights module owns
# the later score-to-weight normalization step.
DEBIASED_EMA_STRATEGY = "debiased_ema"
ROLLING_MA_SOFTMAX_STRATEGY = "rolling_ma_softmax"
LATEST_STRATEGY = "latest"
DEFAULT_SCORE_STRATEGY = ROLLING_MA_SOFTMAX_STRATEGY
DEFAULT_SCORE_WINDOW_SECONDS = 24 * 60 * 60
DEFAULT_SCORE_SOFTMAX_BETA = 10.0
DEFAULT_RAW_HISTORY_MAX_EVENTS = 5000
SCORE_STRATEGIES: dict[str, ScoreStrategy] = {
    DEBIASED_EMA_STRATEGY: ScoreStrategy(
        name=DEBIASED_EMA_STRATEGY,
        update=_ema_update,
        read=_debiased_ema_scores,
        description="Debiased EMA over observed miner rewards.",
    ),
    LATEST_STRATEGY: ScoreStrategy(
        name=LATEST_STRATEGY,
        update=_latest_update,
        read=_raw_scores,
        description="Use only the latest observed reward for each miner.",
    ),
    ROLLING_MA_SOFTMAX_STRATEGY: ScoreStrategy(
        name=ROLLING_MA_SOFTMAX_STRATEGY,
        update=_latest_update,
        read=_rolling_ma_softmax_scores,
        uses_raw_history=True,
        description=(
            "Average raw miner scores over a rolling time window, "
            "then apply softmax."
        ),
    ),
}


def score_strategy(name: str | ScoreStrategy) -> ScoreStrategy:
    if isinstance(name, ScoreStrategy):
        return name
    try:
        return SCORE_STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCORE_STRATEGIES))
        raise ValueError(f"unknown score strategy {name!r}; available strategies: {known}") from exc


def _raw_history_event(
    *,
    timestamp: float,
    uids: np.ndarray,
    raw_scores: np.ndarray,
) -> dict[str, Any]:
    return {
        "timestamp": float(timestamp),
        "uids": uids.astype(np.int64).tolist(),
        "raw_scores": raw_scores.astype(np.float64).tolist(),
    }


def _coerce_raw_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            timestamp = float(entry.get("timestamp", 0.0))
        except (TypeError, ValueError):
            continue
        uids = entry.get("uids", [])
        raw_scores = entry.get("raw_scores", entry.get("scores", []))
        if not isinstance(uids, list) or not isinstance(raw_scores, list):
            continue
        clean_uids: list[int] = []
        clean_scores: list[float] = []
        for uid_value, score_value in zip(uids, raw_scores):
            try:
                uid = int(uid_value)
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            score = float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
            clean_uids.append(uid)
            clean_scores.append(score)
        if clean_uids:
            events.append(
                {
                    "timestamp": timestamp,
                    "uids": clean_uids,
                    "raw_scores": clean_scores,
                }
            )
    return events


class ScoreBoard:
    def __init__(
        self,
        size: int,
        *,
        alpha: float = 0.05,
        strategy: str | ScoreStrategy = DEFAULT_SCORE_STRATEGY,
        window_seconds: float = DEFAULT_SCORE_WINDOW_SECONDS,
        softmax_beta: float = DEFAULT_SCORE_SOFTMAX_BETA,
        raw_history_max_events: int = DEFAULT_RAW_HISTORY_MAX_EVENTS,
    ):
        self.alpha = float(alpha)
        self.strategy = score_strategy(strategy)
        self.ema = np.zeros(size, dtype=np.float64)
        self.counts = np.zeros(size, dtype=np.int64)
        self.window_seconds = float(window_seconds)
        self.softmax_beta = float(softmax_beta)
        self.raw_history_max_events = int(raw_history_max_events)
        self.raw_history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for older callers."""
        return self.raw_history

    @history.setter
    def history(self, value: Any) -> None:
        self.raw_history = _coerce_raw_history(value)

    @property
    def scores(self) -> np.ndarray:
        return self.strategy.read(
            ScoreReadContext(
                score_state=self.ema,
                counts=self.counts,
                alpha=self.alpha,
                raw_history=self.raw_history,
                now=time.time(),
                window_seconds=self.window_seconds,
                softmax_beta=self.softmax_beta,
            )
        )

    def resize(self, size: int) -> None:
        if size == len(self.ema):
            return
        new_ema = np.zeros(size, dtype=np.float64)
        new_counts = np.zeros(size, dtype=np.int64)
        copy_len = min(size, len(self.ema))
        new_ema[:copy_len] = self.ema[:copy_len]
        new_counts[:copy_len] = self.counts[:copy_len]
        self.ema = new_ema
        self.counts = new_counts

    def observe(
        self,
        uids: Iterable[int],
        rewards: Iterable[float],
        *,
        timestamp: float | None = None,
    ) -> None:
        uid_array = np.asarray(list(uids), dtype=np.int64)
        reward_array = np.asarray(list(rewards), dtype=np.float64)
        if len(uid_array) == 0:
            return
        if len(uid_array) != len(reward_array):
            raise ValueError("uids and rewards must have the same length")
        reward_array = np.nan_to_num(
            reward_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        observed_at = float(time.time() if timestamp is None else timestamp)
        self.raw_history.append(
            _raw_history_event(
                timestamp=observed_at,
                uids=uid_array,
                raw_scores=reward_array,
            )
        )
        self._prune_raw_history()
        self.counts[uid_array] += 1
        self.ema[uid_array] = self.strategy.update(
            self.ema[uid_array],
            reward_array,
            self.counts[uid_array],
            self.alpha,
        )

    def _prune_raw_history(self) -> None:
        self.raw_history.sort(key=lambda entry: float(entry.get("timestamp", 0.0)))
        if self.raw_history_max_events > 0:
            self.raw_history = self.raw_history[-self.raw_history_max_events :]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "strategy": self.strategy.name,
                    "score_state": self.ema.tolist(),
                    # Backward-compatible key for older scoreboard files.
                    "ema": self.ema.tolist(),
                    "counts": self.counts.tolist(),
                    "raw_history": self.raw_history,
                    "raw_history_max_events": self.raw_history_max_events,
                    "window_seconds": self.window_seconds,
                    "softmax_beta": self.softmax_beta,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        size: int,
        alpha: float,
        strategy: str | ScoreStrategy = DEFAULT_SCORE_STRATEGY,
        window_seconds: float = DEFAULT_SCORE_WINDOW_SECONDS,
        softmax_beta: float = DEFAULT_SCORE_SOFTMAX_BETA,
        raw_history_max_events: int = DEFAULT_RAW_HISTORY_MAX_EVENTS,
    ) -> "ScoreBoard":
        target = Path(path)
        board = cls(
            size,
            alpha=alpha,
            strategy=strategy,
            window_seconds=window_seconds,
            softmax_beta=softmax_beta,
            raw_history_max_events=raw_history_max_events,
        )
        if not target.exists():
            return board
        data = json.loads(target.read_text())
        board.alpha = float(data.get("alpha", alpha))
        board.window_seconds = float(
            data.get("window_seconds", window_seconds)
        )
        board.softmax_beta = float(data.get("softmax_beta", softmax_beta))
        board.raw_history_max_events = int(
            data.get("raw_history_max_events", raw_history_max_events)
        )
        board.ema = np.asarray(
            data.get("score_state", data.get("ema", [])),
            dtype=np.float64,
        )
        board.counts = np.asarray(data.get("counts", []), dtype=np.int64)
        board.raw_history = _coerce_raw_history(
            data.get("raw_history", data.get("history", []))
        )
        board._prune_raw_history()
        board.resize(size)
        return board
