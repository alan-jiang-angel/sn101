"""Pick the tag set that maximises the validator's actual objective.

Rather than hand-tuning heuristics, we reconstruct the validator's scorer
locally and run our candidate sets through it against a synthetic crowd built
from the LLM ensemble. Whatever wins that simulation is what we submit.

Three findings from reading the scorer drive the design:

1. Validity is deterministic and winnable outright. Any tag whose tokens all
   appear in the post scores lexical_overlap 1.0 -> V = 1.0. We bias hard
   toward such tags.
2. Consensus collapses by a factor of e for every rank you sit away from your
   cluster's centroid, and the centroid is frequency-weighted. Only the modal
   surface form is worth submitting.
3. Aggregation is a MEAN over your tags, not a sum. A weak third tag actively
   lowers your score. But a 1-tag answer collides exactly with other miners far
   more often, and the duplicate penalty punishes large identical groups. We
   resolve that tension by measuring both.
"""

from __future__ import annotations

import itertools
import re
import math
from dataclasses import dataclass, field

import numpy as np

from . import config
from .constraints import PostContext, normalize_tag
from .encoder import CachedEncoder

# ---------------------------------------------------------------------------
# Scorer reconstruction
# ---------------------------------------------------------------------------
try:  # Prefer the validator's own classes -- exactness beats reimplementation.
    from tag101.tasks.sn101_reference.core.scoring.consensus import ConsensusScorer
    from tag101.tasks.sn101_reference.core.scoring.diversity import DiversityScorer
    from tag101.tasks.sn101_reference.core.scoring.validity import ValidityScorer

    _HAVE_REFERENCE = True
except Exception:  # noqa: BLE001 - fall back to the local mirror below
    ConsensusScorer = DiversityScorer = ValidityScorer = None  # type: ignore
    _HAVE_REFERENCE = False


@dataclass
class Scored:
    tags: list[str]
    miner_score: float
    per_tag: list[float] = field(default_factory=list)
    consensus: list[float] = field(default_factory=list)
    validity: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)


class SimulatedValidator:
    """Runs the validator's scoring maths over a simulated response set."""

    def __init__(self, encoder: CachedEncoder, post: str):
        self.encoder = encoder
        self.post = post
        self.ctx = PostContext(post)
        if _HAVE_REFERENCE:
            self._consensus = ConsensusScorer(
                n_tags_per_miner=config.N_TAGS,
                proximity_rank_decay=config.PROXIMITY_RANK_DECAY,
                model=encoder,
            )
            self._validity = ValidityScorer(
                n_tags_per_miner=config.N_TAGS, model=encoder
            )
            self._diversity = DiversityScorer(
                n_tags_per_miner=config.N_TAGS, model=encoder
            )
        else:
            self._consensus = self._validity = self._diversity = None

    # -- public -------------------------------------------------------------
    def score_candidate(
        self, candidate: list[str], crowd: list[list[str]]
    ) -> Scored:
        """Score `candidate` as if it were one more miner in `crowd`."""
        responses = [list(r) for r in crowd] + [list(candidate)]
        idx = len(responses) - 1

        consensus = self._consensus_scores(responses)[idx]
        validity = self._validity_scores(responses)[idx]
        diversity = self._diversity_scores(responses)[idx]

        per_tag = [
            config.CONSENSUS_WEIGHT * c + config.VALIDITY_DIVERSITY_WEIGHT * v * d
            for c, v, d in zip(consensus, validity, diversity)
        ]
        score = float(np.mean(per_tag)) if per_tag else 0.0
        return Scored(
            tags=list(candidate),
            miner_score=score,
            per_tag=per_tag,
            consensus=list(consensus),
            validity=list(validity),
            diversity=list(diversity),
        )

    # -- component scores ---------------------------------------------------
    def _consensus_scores(self, responses: list[list[str]]) -> list[list[float]]:
        if self._consensus is not None:
            return self._consensus.score_from_context(
                self._context(responses)
            )["consensus_scores"]
        return self._local_consensus(responses)

    def _validity_scores(self, responses: list[list[str]]) -> list[list[float]]:
        if self._validity is not None:
            return self._validity.score_from_context(
                self._context(responses)
            )["validity_scores"]
        # Local mirror: the verbatim guarantee is exact; everything else we
        # conservatively treat as the middle tier.
        out = []
        for tags in responses:
            row = []
            for tag in tags:
                floor = self.ctx.validity_floor(tag)
                row.append(floor if floor > 0 else 0.6)
            out.append(row)
        return out

    def _diversity_scores(self, responses: list[list[str]]) -> list[list[float]]:
        if self._diversity is not None:
            return self._diversity.score_from_context(
                self._context(responses)
            )["diversity_scores"]
        return [self._local_diversity(tags) for tags in responses]

    def _context(self, responses: list[list[str]]):
        from tag101.tasks.sn101_reference.core.scoring.preprocessing import (
            build_scoring_context,
        )

        return build_scoring_context(
            post=self.post,
            responses=responses,
            n_tags_per_miner=config.N_TAGS,
        )

    # -- local fallbacks ----------------------------------------------------
    def _local_consensus(self, responses: list[list[str]]) -> list[list[float]]:
        from sklearn.cluster import AgglomerativeClustering

        flat, owner = [], []
        for i, tags in enumerate(responses):
            for tag in tags[: config.N_TAGS]:
                flat.append(normalize_tag(tag))
                owner.append(i)
        if not flat:
            return [[] for _ in responses]

        vecs = self.encoder.encode(flat)
        n_clusters = min(config.N_CLUSTERS, len(set(flat)))
        if len(flat) == 1 or n_clusters <= 1:
            labels = np.zeros(len(flat), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            ).fit_predict(vecs)

        by_cluster: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            by_cluster.setdefault(int(lab), []).append(i)

        scores = [0.0] * len(flat)
        n_miners = max(len(responses), 1)
        for _, idxs in by_cluster.items():
            support = len({owner[i] for i in idxs}) / n_miners
            centroid = vecs[idxs].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            dist: dict[str, float] = {}
            for i in idxs:
                dist[flat[i]] = float(1.0 - np.dot(vecs[i], centroid))
            prox: dict[str, float] = {}
            for tag, d in dist.items():
                rank = 1 + sum(
                    1
                    for other in dist.values()
                    if other < d and not math.isclose(other, d, abs_tol=1e-9)
                )
                prox[tag] = math.exp(-config.PROXIMITY_RANK_DECAY * (rank - 1))
            for i in idxs:
                scores[i] = float(np.clip(support * prox[flat[i]], 0.0, 1.0))

        out, cursor = [], 0
        for tags in responses:
            count = len(tags[: config.N_TAGS])
            out.append(scores[cursor : cursor + count])
            cursor += count
        return out

    def _local_diversity(self, tags: list[str]) -> list[float]:
        tags = tags[: config.N_TAGS]
        if len(tags) <= 1:
            return [1.0] * len(tags)
        vecs = self.encoder.encode(tags)
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, -1.0)
        out = []
        for i in range(len(tags)):
            m = float(np.clip(np.max(sim[i]), 0.0, 1.0))
            out.append(_novelty(m))
        return out


def _novelty(similarity: float) -> float:
    low, high = config.DIVERSE_SIM_THRESHOLD, config.DUPLICATE_SIM_THRESHOLD
    if similarity <= low:
        return 1.0
    if similarity >= high:
        return 0.0
    return 1.0 - ((similarity - low) / (high - low))


# ---------------------------------------------------------------------------
# Candidate assembly and search
# ---------------------------------------------------------------------------
def build_candidates(
    ctx: PostContext,
    crowd_ranked: list[tuple[str, float]],
    limit: int,
) -> list[tuple[str, float]]:
    """Merge LLM candidates with guaranteed-validity n-grams from the post.

    Each candidate carries a prior = ensemble frequency, plus a small bonus if
    it is verbatim (which locks V=1.0 and is therefore worth more than its raw
    popularity suggests).
    """
    scored: dict[str, float] = {}
    for tag, freq in crowd_ranked:
        prior = freq
        if ctx.validity_floor(tag) >= 1.0:
            prior += config.VERBATIM_BONUS
        scored[tag] = max(scored.get(tag, 0.0), prior)

    # Seed with salient post n-grams so we always have safe candidates even if
    # the LLM phase timed out entirely.
    for tag in _salient_ngrams(ctx, limit=limit * 2):
        if tag not in scored:
            scored[tag] = config.VERBATIM_BONUS

    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return ranked[:limit]


def _salient_ngrams(ctx: PostContext, limit: int) -> list[str]:
    """Cheap salience ranking over verbatim n-grams.

    Every candidate here is already verbatim, so validity is locked at 1.0 no
    matter how long it is. Length therefore buys nothing and costs consensus:
    rank is keyed on the exact string, and there are far more ways to phrase a
    3-word tag than a 1-word tag, so long tags fragment the crowd's vote and
    rarely end up modal. This ranking is deliberately biased SHORT and toward
    proper nouns, which are the surface forms the field converges on.

    This is the offline safety net, not the primary path.
    """
    stop = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
        "we", "i", "you", "they", "he", "she", "at", "by", "from", "as", "so",
        "just", "now", "new", "our", "your", "my", "has", "have", "had", "will",
        "can", "not", "no", "if", "than", "then", "there", "here", "what",
        "how", "why", "all", "more", "out", "up", "about", "into", "over",
    }
    # Verb-led fragments ("benchmarks show", "shipped gpt-5") read as sentence
    # slices rather than tags, and essentially never appear in another miner's
    # answer -- which means near-zero consensus.
    verbish = {
        "shipped", "show", "shows", "showed", "drop", "drops", "dropped",
        "said", "says", "say", "get", "gets", "got", "make", "makes", "made",
        "use", "uses", "used", "add", "adds", "added", "found", "find",
        "launch", "launched", "release", "released", "announce", "announced",
        "build", "built", "run", "runs", "think", "know", "want", "need",
        "look", "looks", "going", "coming", "jump", "jumps", "recover",
    }
    # Generic modifiers that are verbatim and short but carry no topic -- the
    # exact failure mode of a naive "prefer short tags" rule.
    filler = {
        "hot", "bad", "good", "big", "small", "great", "best", "worst", "old",
        "take", "thing", "things", "stuff", "way", "ways", "lot", "bit",
        "kind", "sort", "part", "time", "day", "year", "people", "guy",
        "really", "very", "much", "many", "most", "some", "any", "other",
        "first", "last", "next", "own", "same", "still", "even", "also",
    }
    raw_words = ctx.post.split()
    caps = set()
    for word in raw_words:
        cleaned = word.strip(".,!?;:\"'()[]{}")
        cleaned = re.sub(r"['\u2019]s$", "", cleaned)
        if cleaned[:1].isupper():
            caps.add(cleaned.lower())

    # Prefer the units the validator itself treats as meaningful spans.
    entities = set(_entity_candidates(ctx))

    out: list[tuple[float, str]] = []
    for gram in ctx.ngram_candidates(max_n=3):
        words = gram.split()
        if all(w in stop or w in filler for w in words):
            continue
        if words[0] in stop or words[-1] in stop:
            continue
        if words[0] in verbish or words[-1] in verbish:
            continue

        score = 0.0
        # Short is better once validity is guaranteed, but a two-word noun
        # phrase ("sparse autoencoders") beats a bare modifier ("sparse").
        score += {1: 1.6, 2: 1.4, 3: 0.4}.get(len(words), 0.0)
        score += 1.5 * sum(1 for w in words if w in caps)   # proper nouns
        score += 2.0 if gram in entities else 0.0           # validator spans
        score -= 0.6 * sum(1 for w in words if w in stop)
        score -= 1.0 * sum(1 for w in words if w in verbish)
        score -= 1.8 * sum(1 for w in words if w in filler)
        # Longer words are rarer and more topical: a crude but effective
        # stand-in for POS tagging when spaCy is unavailable.
        score += 0.12 * sum(max(0, len(w) - 5) for w in words)
        out.append((score, gram))

    out.sort(key=lambda kv: (-kv[0], len(kv[1])))
    return [g for _, g in out[:limit]]


def _entity_candidates(ctx: PostContext) -> list[str]:
    """Entities and noun chunks, using the same spaCy model the validator uses
    to build scoring spans. Returns [] when spaCy is unavailable."""
    try:
        from tag101.tasks.sn101_reference.core.scoring.preprocessing import (
            extract_entity_and_noun_chunk_spans,
        )
    except Exception:  # noqa: BLE001
        return []
    try:
        spans = extract_entity_and_noun_chunk_spans(ctx.post)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for span in spans:
        tag = normalize_tag(span)
        if tag and ctx.validity_floor(tag) >= 1.0:
            out.append(tag)
    return out


def estimate_duplicate_decay(
    tags: list[str],
    crowd_samples: list[list[str]],
    n_miners: int | None = None,
) -> float:
    """Estimate the duplicate-set penalty this answer would attract.

    The validator groups miners by their exact normalised tag set and scales
    each group down by 1/(1+exp(k*(n-c))). We estimate the collision rate as
    the fraction of sampled crowd responses that contain ALL of our tags --
    a set that every third miner would also produce is one we should avoid,
    however individually popular its members are.

    This is what stops the search from converging on a single ultra-popular
    tag, which maximises the raw mean but lands in the largest possible
    identical group.
    """
    n_miners = n_miners or config.ASSUMED_MINER_COUNT
    if not tags:
        return 1.0
    if not crowd_samples:
        # No evidence either way; assume mild collision pressure that grows as
        # the answer gets shorter, since short sets are combinatorially easier
        # to match.
        assumed = {1: 0.25, 2: 0.06, 3: 0.02}.get(len(tags), 0.02)
        collision = assumed
    else:
        wanted = set(tags)
        hits = sum(1 for sample in crowd_samples if wanted <= set(sample))
        collision = hits / len(crowd_samples)

    expected_group = 1.0 + max(0, n_miners - 1) * collision
    exponent = config.DUPLICATE_PENALTY_K * (
        expected_group - config.DUPLICATE_PENALTY_C
    )
    exponent = max(-60.0, min(60.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def select_tags(
    post: str,
    encoder: CachedEncoder,
    crowd_ranked: list[tuple[str, float]],
    crowd_samples: list[list[str]],
    top_k: int | None = None,
    time_budget: float | None = None,
) -> Scored:
    """Search tag sets and return the one with the best expected reward.

    The objective is the validator's adjusted miner score:
        mean(tag_scores) * estimated_duplicate_decay
    """
    import time as _time

    deadline = _time.perf_counter() + float(
        time_budget or config.SELECT_TIME_BUDGET
    )
    ctx = PostContext(post)
    sim = SimulatedValidator(encoder, post)
    top_k = top_k or config.SELECT_TOP_K

    candidates = build_candidates(ctx, crowd_ranked, limit=top_k)
    pool = [tag for tag, _ in candidates]
    if not pool:
        return Scored(tags=[], miner_score=0.0)

    # Without a crowd we cannot estimate consensus; the objective degrades
    # gracefully to validity x diversity, which still yields a sane answer.
    crowd = crowd_samples if crowd_samples else []

    def evaluate(tags: list[str]) -> Scored | None:
        if len(tags) > 1:
            # Hard diversity gate: a pair at/above the duplicate threshold
            # zeroes that tag's V*D term outright.
            if encoder.max_pairwise_similarity(tags) >= config.DUPLICATE_SIM_THRESHOLD:
                return None
        result = sim.score_candidate(tags, crowd)
        decay = estimate_duplicate_decay(tags, crowd)
        result.miner_score *= decay
        return result

    best: Scored | None = None
    # Search widest sets first so that if we run out of time we still hold a
    # 3-tag answer, which is the safer default.
    for k in (3, 2, 1):
        if len(pool) < k:
            continue
        for combo in itertools.combinations(pool, k):
            if _time.perf_counter() > deadline and best is not None:
                return best
            result = evaluate(list(combo))
            if result is None:
                continue
            if best is None:
                best = result
                continue
            # Tie-break toward more tags: equal expected score with more tags
            # is strictly safer against both collision and single-tag variance.
            margin = config.MARGINAL_TAG_EPSILON if len(result.tags) < len(best.tags) else 0.0
            if result.miner_score > best.miner_score + margin:
                best = result

    if best is None:
        return Scored(tags=pool[: config.N_TAGS], miner_score=0.0)
    return best
