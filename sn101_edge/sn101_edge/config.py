"""Runtime configuration for the SN101 edge miner.

Everything is env-driven so the miner can be tuned without a rebuild. Defaults
are chosen to be safe on a 10s wire timeout, which is the worst case the task
server can hand us (TaskLease.time_limit defaults to 10.0).
"""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Scoring constants mirrored from the validator ---------------------------
# Kept here so the selector can score candidates without importing sklearn.
N_TAGS = 3
CONSENSUS_WEIGHT = 0.60
VALIDITY_DIVERSITY_WEIGHT = 0.40
DIVERSE_SIM_THRESHOLD = 0.55      # <= this  -> diversity 1.0
DUPLICATE_SIM_THRESHOLD = 0.85    # >= this  -> diversity 0.0
N_CLUSTERS = N_TAGS * 2           # validator clusters into exactly 6
PROXIMITY_RANK_DECAY = 1.0

# --- Embedding model ---------------------------------------------------------
EMBED_MODEL = os.getenv("SN101_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- LLM ensemble ------------------------------------------------------------
# OpenRouter is preferred; the repo's OPENAI_* vars are honoured as a fallback
# so this drops into an existing deployment without new secrets.
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv(
    "SN101_LLM_BASE_URL",
    "https://openrouter.ai/api/v1" if OPENROUTER_KEY else os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    ),
)
LLM_API_KEY = OPENROUTER_KEY or OPENAI_KEY

# The crowd overwhelmingly runs the reference miner on gpt-4o-mini, so that
# model is the single most valuable ensemble member: it samples the same
# distribution the consensus score is computed over. The others add coverage.
_DEFAULT_MODELS = "openai/gpt-4o-mini,google/gemini-2.0-flash-001,openai/gpt-4o-mini"
ENSEMBLE_MODELS = [
    m.strip() for m in os.getenv("SN101_MODELS", _DEFAULT_MODELS).split(",") if m.strip()
]
ENSEMBLE_TEMPERATURE = _f("SN101_TEMPERATURE", 0.9)
ENSEMBLE_SAMPLES_PER_MODEL = _i("SN101_SAMPLES_PER_MODEL", 3)
LLM_MAX_CONCURRENCY = _i("SN101_LLM_CONCURRENCY", 8)

# --- Deadlines ---------------------------------------------------------------
# Fraction of the validator's wire timeout we allow the LLM phase to consume.
# The remainder covers local embedding, selection, and JSON serialisation.
LLM_BUDGET_FRACTION = _f("SN101_LLM_BUDGET_FRACTION", 0.45)
LOCAL_BUDGET_FRACTION = _f("SN101_LOCAL_BUDGET_FRACTION", 0.25)
MIN_LLM_BUDGET = _f("SN101_MIN_LLM_BUDGET", 1.5)
FALLBACK_TIMEOUT_GUESS = _f("SN101_FALLBACK_TIMEOUT", 10.0)

# --- Caching -----------------------------------------------------------------
# Multiple validators send the SAME post within a short window. Caching turns
# every validator after the first into an instant, zero-cost response and keeps
# our answer identical across them.
CACHE_TTL_SECONDS = _f("SN101_CACHE_TTL", 3600.0)
CACHE_MAX_ENTRIES = _i("SN101_CACHE_MAX", 2048)

# --- Selection ---------------------------------------------------------------
# How many top candidates feed the triple search. K=6 -> 20 triples, which is
# fast with a cached encoder and still covers the plausible answer space.
SELECT_TOP_K = _i("SN101_SELECT_TOP_K", 6)
# Wall-clock ceiling for the combinatorial search itself.
SELECT_TIME_BUDGET = _f("SN101_SELECT_BUDGET", 2.5)
# Only keep a marginal tag if its predicted score beats the running mean by
# this margin. Aggregation is a MEAN, so a weak third tag actively hurts.
MARGINAL_TAG_EPSILON = _f("SN101_MARGINAL_EPSILON", 0.02)

# --- Duplicate penalty modelling ---------------------------------------------
# The production validator scores with k=0.06, c=80 (see tasks/sn101.py), not
# the README's k=0.1, c=50. Decay = 1/(1+exp(k*(n-c))) where n is the number of
# miners submitting the IDENTICAL normalised tag set. A popular one-tag answer
# is the worst case here: if 80+ miners all answer ["openai"], every one of
# them is halved. We estimate n from crowd concentration and fold it into the
# objective so the search naturally avoids crowded sets.
DUPLICATE_PENALTY_K = _f("SN101_DUP_K", 0.06)
DUPLICATE_PENALTY_C = _f("SN101_DUP_C", 80.0)
ASSUMED_MINER_COUNT = _i("SN101_ASSUMED_MINERS", 200)
# Prefer tags that are literal token-subsets of the post: those are guaranteed
# validity 1.0 via the lexical-overlap path.
VERBATIM_BONUS = _f("SN101_VERBATIM_BONUS", 0.03)

DEBUG = _b("SN101_DEBUG", False)
