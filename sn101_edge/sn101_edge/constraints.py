"""Bit-exact mirror of the validator's deterministic validity rules.

The validator computes:

    base_score  = max(sim_post, sim_span, lexical_overlap)
    raw         = base_score * format_score
    V           = tier(raw)   in {0, 0.3, 0.6, 1.0}

`lexical_overlap` is the fraction of the tag's tokens that appear in the token
set of SOME span, and the full post is always a span. So any tag whose tokens
are all drawn from the post gets lexical_overlap == 1.0, hence base_score 1.0,
hence V == 1.0 outright -- no embedding required, no uncertainty.

That makes validity a solved sub-problem: constrain candidates to token-subsets
of the post and pass the format gate, and 40% of the reward formula is locked
at maximum. This module implements those two checks exactly as the validator
does, including its idiosyncratic stemmer and its token counter (which splits
on non-alphanumerics, so "gpt-5" counts as TWO tokens).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Copied verbatim from the validator's ValidityScorer.
_URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(com|org|net|io|ai|co)\b)",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_NUMERIC_ONLY = re.compile(r"[\d\s.,:/%+\-]+")
_PUNCT_ONLY = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_ALLOWED_SPECIALS = {"-", "_", "&", "/", "+"}

MIN_TOKENS = 1
MAX_TOKENS = 5


def normalize_tag(tag: str) -> str:
    """Matches preprocessing.normalize_tag."""
    return re.sub(r"\s+", " ", tag.strip().lower())


def _normalize_token(token: str) -> str:
    """The validator's stemmer. Applied to both tags and spans, so it is
    self-consistent: 'benchmarks' and 'benchmark' collapse to the same key."""
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_normalize_token(t) for t in _TOKEN_PATTERN.findall(text.lower())]


def token_count(text: str) -> int:
    """Raw token count as the format gate sees it.

    Note this splits on every non-alphanumeric, so hyphens and embedded digits
    inflate the count: 'gpt-5 turbo' is 3 tokens, not 2, and
    'openai gpt-5 turbo benchmarks pricing' is 6 -> instant format failure.
    """
    return len(_TOKEN_PATTERN.findall(text.lower()))


def format_score(tag: str) -> float:
    """Mirror of ValidityScorer._format_score. Returns 1.0 or 0.0."""
    normalized = tag.strip()
    if not normalized:
        return 0.0
    if _URL_PATTERN.search(normalized):
        return 0.0

    count = token_count(normalized)
    if count < MIN_TOKENS or count > MAX_TOKENS:
        return 0.0

    if _NUMERIC_ONLY.fullmatch(normalized):
        return 0.0
    if _PUNCT_ONLY.fullmatch(normalized):
        return 0.0
    if _is_emoji_only(normalized):
        return 0.0
    if _too_many_specials(normalized):
        return 0.0
    return 1.0


def _is_emoji_only(text: str) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    if not compact:
        return False
    if any(ch.isalnum() for ch in compact):
        return False
    return all(unicodedata.category(ch).startswith(("S", "P")) for ch in compact)


def _too_many_specials(text: str) -> bool:
    compact = [ch for ch in text if not ch.isspace()]
    if not compact:
        return False
    special = sum(
        1 for ch in compact if not ch.isalnum() and ch not in _ALLOWED_SPECIALS
    )
    return (special / len(compact)) > 0.3


def tier_validity(raw: float) -> float:
    if raw < 0.15:
        return 0.0
    if raw < 0.35:
        return 0.3
    if raw < 0.65:
        return 0.6
    return 1.0


class PostContext:
    """Precomputed view of one post: its token set and its n-gram candidates."""

    def __init__(self, post: str):
        self.post = post
        self.token_set = set(tokenize(post))
        self._words = self._word_sequence(post)

    @staticmethod
    def _word_sequence(post: str) -> list[str]:
        # Keep sentence boundaries so we do not build n-grams that straddle
        # unrelated clauses -- those read as noise and lose consensus.
        chunks = re.split(r"[.!?\n]+|[,;:]", post)
        seq: list[str] = []
        for chunk in chunks:
            words = [w for w in re.split(r"\s+", chunk.strip()) if w]
            if words:
                seq.append("\u0000")  # boundary marker
                seq.extend(words)
        return seq

    def is_verbatim(self, tag: str) -> bool:
        """True iff every token of the tag appears in the post's token set.

        This is precisely the condition for lexical_overlap == 1.0.
        """
        toks = tokenize(tag)
        if not toks:
            return False
        return all(t in self.token_set for t in toks)

    def validity_floor(self, tag: str) -> float:
        """Guaranteed validity ignoring any embedding contribution.

        A verbatim tag that passes the format gate is a hard 1.0. Anything else
        we treat as unknown here (0.0) and let the embedding path decide.
        """
        if format_score(tag) <= 0.0:
            return 0.0
        if self.is_verbatim(tag):
            return 1.0
        return 0.0

    def ngram_candidates(self, max_n: int = 4) -> list[str]:
        """All contiguous 1..max_n word n-grams from the post that survive the
        format gate. These are the raw material for guaranteed-validity tags."""
        out: list[str] = []
        seen: set[str] = set()
        words = self._words
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                window = words[i : i + n]
                if any(w == "\u0000" for w in window):
                    continue
                phrase = _clean_phrase(" ".join(window))
                if not phrase:
                    continue
                norm = normalize_tag(phrase)
                if norm in seen:
                    continue
                if format_score(norm) <= 0.0:
                    continue
                if not self.is_verbatim(norm):
                    continue
                seen.add(norm)
                out.append(norm)
        return out


def _clean_phrase(text: str) -> str:
    """Strip decoration the validator would treat as special-symbol noise.

    Possessives are dropped ("anthropic's" -> "anthropic") because the bare
    form is both a more conventional tag -- and therefore more likely to be the
    crowd's modal surface form -- and still verbatim, since the apostrophe is
    not a token boundary the validator's tokenizer cares about.
    """
    text = text.strip()
    text = re.sub(r"^[^\w#@]+|[^\w%]+$", "", text)
    text = text.lstrip("#@")
    text = re.sub(r"['\u2019]s\b", "", text)
    text = re.sub(r"['\u2019]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def repair_tag(tag: str, ctx: PostContext) -> str | None:
    """Best-effort coercion of a raw LLM tag into something that scores.

    Returns None if the tag is unsalvageable. The main repairs are dropping
    decoration and trimming over-long tags down to the 5-token format ceiling.
    """
    if not isinstance(tag, str):
        return None
    candidate = normalize_tag(_clean_phrase(tag))
    if not candidate:
        return None
    if _URL_PATTERN.search(candidate):
        return None

    if token_count(candidate) > MAX_TOKENS:
        # Trim from the right; leading words usually carry the head noun.
        words = candidate.split()
        while words and token_count(" ".join(words)) > MAX_TOKENS:
            words.pop()
        candidate = " ".join(words)

    if not candidate or format_score(candidate) <= 0.0:
        return None
    return candidate


def dedupe_preserving_order(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        norm = normalize_tag(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out
