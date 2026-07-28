"""Orchestration: turn a post into three tags, on time, every single time.

Reliability is worth more than cleverness on this subnet. The default score
strategy is `rolling_ma_softmax` over a 24h window, and it computes each
miner's average as `sum / max_count` -- where max_count is the highest number
of observations ANY miner has. A missed or timed-out task is therefore a hard
zero that dilutes your average for a full day, and with softmax_beta = 10 a
0.1 drop in average score costs you a factor of e in emissions.

So: every path through this module returns a valid answer inside the deadline.
The LLM ensemble is an optimisation layered on top of a local extractive
tagger that always works, never blocks, and needs no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

from . import config
from .constraints import PostContext
from .crowd import CrowdEstimate, CrowdSampler
from .encoder import get_encoder, warm_in_background
from .selector import Scored, select_tags

_LOG_PREFIX = "SN101_EDGE"


def _log(message: str) -> None:
    if config.DEBUG:
        print(f"[{_LOG_PREFIX}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Background event loop
# ---------------------------------------------------------------------------
# solve_problem is called synchronously from inside the miner's running event
# loop, so we cannot use asyncio.run(). A dedicated loop on its own thread lets
# us drive concurrent HTTP without touching the server's loop.
class _BackgroundLoop:
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            with self._lock:
                if self._loop is None:
                    loop = asyncio.new_event_loop()
                    thread = threading.Thread(
                        target=loop.run_forever, daemon=True,
                        name="sn101-edge-loop",
                    )
                    thread.start()
                    self._loop, self._thread = loop, thread
        return self._loop

    def run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop())
        try:
            return future.result(timeout=timeout)
        except Exception:  # noqa: BLE001 - includes TimeoutError
            future.cancel()
            return None


_BG = _BackgroundLoop()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    tags: list[str]
    created: float


class _PostCache:
    """Validators query the same post independently, often within seconds.

    Caching makes every validator after the first free and instantaneous, and
    guarantees we submit an identical answer to all of them.
    """

    def __init__(self):
        self._data: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(post: str) -> str:
        return hashlib.sha1(post.strip().lower().encode("utf-8")).hexdigest()

    def get(self, post: str) -> list[str] | None:
        k = self.key(post)
        now = time.time()
        with self._lock:
            entry = self._data.get(k)
            if entry is None:
                return None
            if now - entry.created > config.CACHE_TTL_SECONDS:
                self._data.pop(k, None)
                return None
            return list(entry.tags)

    def put(self, post: str, tags: list[str]) -> None:
        if not tags:
            return
        with self._lock:
            if len(self._data) >= config.CACHE_MAX_ENTRIES:
                oldest = min(self._data.items(), key=lambda kv: kv[1].created)[0]
                self._data.pop(oldest, None)
            self._data[self.key(post)] = _CacheEntry(list(tags), time.time())


_CACHE = _PostCache()


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------
class EdgeTagger:
    def __init__(self, sampler: CrowdSampler | None = None):
        self.sampler = sampler or CrowdSampler()
        warm_in_background()

    def generate_tags(self, post: str, timeout: float | None = None) -> list[str]:
        started = time.perf_counter()
        post = (post or "").strip()
        if not post:
            return []

        cached = _CACHE.get(post)
        if cached is not None:
            _log(f"cache hit -> {cached}")
            return cached

        budget = float(timeout or config.FALLBACK_TIMEOUT_GUESS)
        llm_budget = max(0.0, min(budget * config.LLM_BUDGET_FRACTION,
                                  budget - 1.0))
        if llm_budget < config.MIN_LLM_BUDGET:
            llm_budget = 0.0

        ctx = PostContext(post)
        encoder = get_encoder()

        # Phase 1 -- sample the crowd, bounded.
        estimate = CrowdEstimate()
        if llm_budget > 0 and self.sampler.enabled:
            result = _BG.run(
                self.sampler.sample(post, ctx, llm_budget),
                timeout=llm_budget + 0.5,
            )
            if isinstance(result, CrowdEstimate):
                estimate = result
        _log(
            f"crowd samples={estimate.n_samples} "
            f"errors={estimate.errors[:3]} "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )

        # Phase 2 -- optimise against the simulated validator, bounded.
        remaining = budget - (time.perf_counter() - started)
        try:
            if remaining < 0.4:
                raise TimeoutError("no budget left for selection")
            chosen = select_tags(
                post=post,
                encoder=encoder,
                crowd_ranked=estimate.ranked(),
                crowd_samples=estimate.as_simulated_miners(),
                time_budget=min(remaining - 0.3, config.SELECT_TIME_BUDGET),
            )
            tags = chosen.tags
            _log(
                f"selected={tags} score={chosen.miner_score:.4f} "
                f"C={[round(x, 2) for x in chosen.consensus]} "
                f"V={chosen.validity} D={[round(x, 2) for x in chosen.diversity]}"
            )
        except Exception as exc:  # noqa: BLE001 - never fail a task
            _log(f"selection failed ({type(exc).__name__}: {exc}); using fallback")
            tags = []

        # Phase 3 -- guarantee an answer.
        if not tags:
            tags = self.local_fallback(post, estimate)

        tags = tags[: config.N_TAGS]
        _CACHE.put(post, tags)
        _log(f"final={tags} total={time.perf_counter() - started:.2f}s")
        return tags

    # -- fallbacks ----------------------------------------------------------
    def local_fallback(self, post: str, estimate: CrowdEstimate | None = None) -> list[str]:
        """Network-free, embedding-free answer. Always returns something valid.

        Prefers the ensemble's most frequent tags if we got any, otherwise
        salient verbatim n-grams -- all of which carry a guaranteed V = 1.0.
        """
        from .selector import _salient_ngrams

        ctx = PostContext(post)
        picks: list[str] = []

        if estimate is not None and estimate.n_samples > 0:
            for tag, _freq in estimate.ranked(limit=12):
                if ctx.validity_floor(tag) >= 1.0 and tag not in picks:
                    picks.append(tag)
                if len(picks) >= config.N_TAGS:
                    break

        if len(picks) < config.N_TAGS:
            for gram in _salient_ngrams(ctx, limit=24):
                if gram in picks:
                    continue
                # Crude lexical de-duplication so the three picks do not share
                # words -- a decent proxy for the embedding diversity gate.
                words = set(gram.split())
                if any(words & set(p.split()) for p in picks):
                    continue
                picks.append(gram)
                if len(picks) >= config.N_TAGS:
                    break

        if not picks:
            words = [w for w in post.lower().split() if w.isalnum()]
            picks = words[: config.N_TAGS] or ["post"]
        return picks[: config.N_TAGS]


_TAGGER: EdgeTagger | None = None
_TAGGER_LOCK = threading.Lock()


def get_tagger() -> EdgeTagger:
    global _TAGGER
    if _TAGGER is None:
        with _TAGGER_LOCK:
            if _TAGGER is None:
                _TAGGER = EdgeTagger()
    return _TAGGER


def generate_tags(post: str, timeout: float | None = None) -> list[str]:
    return get_tagger().generate_tags(post, timeout=timeout)


def solve_envelope(envelope: Any) -> dict[str, list[str]]:
    """Adapter used by the task handler."""
    payload = dict(getattr(envelope, "payload", {}) or {})
    post = str(payload.get("text", ""))
    timeout = float(
        getattr(envelope, "timeout", None)
        or getattr(envelope, "time_limit", None)
        or config.FALLBACK_TIMEOUT_GUESS
    )
    return {"tags": generate_tags(post, timeout=timeout)}
