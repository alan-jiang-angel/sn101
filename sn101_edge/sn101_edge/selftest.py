"""Offline self-test. Run before deploying:  python -m sn101_edge.selftest

Exercises the deterministic guarantees (format gate, verbatim validity,
n-gram generation, fallback) with a stub encoder so it needs no network and no
model checkpoint. Add --live to additionally test the real embedding model and,
if a key is configured, the LLM ensemble.
"""

from __future__ import annotations

import sys
import time

from . import config
from .constraints import (
    PostContext,
    format_score,
    repair_tag,
    token_count,
    tokenize,
)
from .crowd import CrowdEstimate, _parse_tags
from .encoder import CachedEncoder, HashEncoder, set_encoder
from .selector import select_tags
from .tagger import EdgeTagger

POSTS = [
    "OpenAI just shipped GPT-5 Turbo. Benchmarks show a big jump in agentic "
    "coding, and pricing drops 40%.",
    "Anthropic's new interpretability work found that sparse autoencoders "
    "recover monosemantic features from Claude's residual stream.",
    "hot take: most RAG pipelines are just bad search with extra steps",
]

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_format_gate() -> None:
    print("\nFormat gate (mirrors ValidityScorer._format_score)")
    check("single word accepted", format_score("openai") == 1.0)
    check("hyphen+digit counts as 3 tokens",
          token_count("gpt-5 turbo") == 3, "gpt|5|turbo")
    check("5 tokens accepted", format_score("a b c d e") == 1.0)
    check("6 tokens rejected", format_score("a b c d e f") == 0.0)
    check("url rejected", format_score("https://openai.com") == 0.0)
    check("bare domain rejected", format_score("openai.com") == 0.0)
    check("numeric only rejected", format_score("40") == 0.0)
    check("punctuation only rejected", format_score("!!!") == 0.0)
    check("symbol spam rejected", format_score("openai!!!@#$") == 0.0)
    check("allowed specials kept", format_score("open-source ai") == 1.0)


def test_stemmer_symmetry() -> None:
    print("\nStemmer symmetry (tags and spans share the same normaliser)")
    check("plural collapses", tokenize("benchmarks") == tokenize("benchmark"))
    check("ies -> y", tokenize("companies") == tokenize("company"))
    check("short words untouched", tokenize("ai") == ["ai"])


def test_verbatim_guarantee() -> None:
    print("\nVerbatim guarantee (lexical_overlap == 1.0 -> V == 1.0)")
    ctx = PostContext(POSTS[0])
    check("verbatim phrase", ctx.validity_floor("agentic coding") == 1.0)
    check("reordered tokens still verbatim",
          ctx.validity_floor("pricing benchmarks") == 1.0)
    check("singular of post plural", ctx.validity_floor("benchmark") == 1.0)
    check("absent term not guaranteed",
          ctx.validity_floor("artificial intelligence") == 0.0)
    check("verbatim but 6 tokens fails format",
          ctx.validity_floor("openai gpt-5 turbo benchmarks pricing") == 0.0)


def test_ngrams() -> None:
    print("\nN-gram candidate generation")
    for post in POSTS:
        ctx = PostContext(post)
        grams = ctx.ngram_candidates(max_n=3)
        all_valid = all(ctx.validity_floor(g) == 1.0 for g in grams)
        check(f"all n-grams score V=1.0 ({len(grams)} from post[:30]={post[:30]!r})",
              all_valid and len(grams) > 0)


def test_repair() -> None:
    print("\nTag repair")
    ctx = PostContext(POSTS[0])
    check("hashtag stripped", repair_tag("#OpenAI", ctx) == "openai")
    check("case normalised", repair_tag("  Agentic Coding ", ctx) == "agentic coding")
    check("url dropped", repair_tag("https://x.com/a", ctx) is None)
    long_tag = repair_tag("one two three four five six seven", ctx)
    check("over-long trimmed to <=5 tokens",
          long_tag is not None and token_count(long_tag) <= 5, str(long_tag))


def test_parser() -> None:
    print("\nLLM response parsing")
    check("plain json", _parse_tags('["a","b","c"]') == ["a", "b", "c"])
    check("fenced json", _parse_tags('```json\n["a","b"]\n```') == ["a", "b"])
    check("prose-wrapped json",
          _parse_tags('Here you go: ["a","b"] hope that helps') == ["a", "b"])
    check("comma fallback", _parse_tags("a, b, c") == ["a", " b", " c"])


def test_selection() -> None:
    print("\nSelection against a simulated crowd (stub encoder)")
    set_encoder(CachedEncoder(model=HashEncoder(), dim=64))
    from .encoder import get_encoder

    encoder = get_encoder()
    post = POSTS[0]

    crowd_samples = [
        ["openai", "gpt-5 turbo", "agentic coding"],
        ["openai", "gpt-5 turbo", "pricing"],
        ["openai", "benchmarks", "agentic coding"],
        ["open ai", "gpt5", "coding"],
        ["openai", "gpt-5 turbo", "benchmarks"],
    ]
    counts: dict[str, int] = {}
    for sample in crowd_samples:
        for tag in sample:
            counts[tag] = counts.get(tag, 0) + 1
    ranked = sorted(
        ((t, c / len(crowd_samples)) for t, c in counts.items()),
        key=lambda kv: -kv[1],
    )

    started = time.perf_counter()
    chosen = select_tags(post, encoder, ranked, crowd_samples)
    elapsed = time.perf_counter() - started

    print(f"       chose {chosen.tags} score={chosen.miner_score:.4f} "
          f"in {elapsed * 1000:.0f}ms")
    check("returned 1-3 tags", 1 <= len(chosen.tags) <= 3, str(chosen.tags))
    check("selection is fast", elapsed < 5.0, f"{elapsed:.2f}s")
    ctx = PostContext(post)
    check("every chosen tag passes format",
          all(format_score(t) == 1.0 for t in chosen.tags))
    check("picked the modal tag", "openai" in chosen.tags, str(chosen.tags))
    check("no exact duplicates", len(set(chosen.tags)) == len(chosen.tags))


def test_fallback() -> None:
    print("\nLocal fallback (no network, no LLM)")
    tagger = EdgeTagger.__new__(EdgeTagger)  # skip __init__ / model warmup
    for post in POSTS:
        tags = tagger.local_fallback(post, CrowdEstimate())
        ctx = PostContext(post)
        ok = (
            1 <= len(tags) <= config.N_TAGS
            and all(format_score(t) == 1.0 for t in tags)
            and all(ctx.validity_floor(t) == 1.0 for t in tags)
        )
        check(f"fallback valid for post[:30]={post[:30]!r}", ok, str(tags))


def test_deadline() -> None:
    print("\nDeadline discipline")
    set_encoder(CachedEncoder(model=HashEncoder(), dim=64))
    tagger = EdgeTagger.__new__(EdgeTagger)
    from .crowd import CrowdSampler

    tagger.sampler = CrowdSampler(api_key="")  # forces the no-LLM path
    for budget in (2.0, 10.0):
        started = time.perf_counter()
        tags = tagger.generate_tags(POSTS[1] + f" #{budget}", timeout=budget)
        elapsed = time.perf_counter() - started
        check(f"answered within {budget}s budget", elapsed < budget,
              f"{elapsed:.2f}s -> {tags}")


def test_live() -> None:
    print("\nLIVE: real embedding model")
    set_encoder(CachedEncoder())
    from .encoder import get_encoder

    encoder = get_encoder()
    started = time.perf_counter()
    encoder.warm()
    print(f"       model loaded in {time.perf_counter() - started:.1f}s")

    pairs = [
        ("openai", "open ai"),
        ("openai", "gpt-5 turbo"),
        ("agentic coding", "software engineering"),
        ("pricing", "benchmarks"),
    ]
    print("       pairwise similarity (diversity gate at "
          f"{config.DIVERSE_SIM_THRESHOLD} / {config.DUPLICATE_SIM_THRESHOLD}):")
    for a, b in pairs:
        vectors = encoder.encode([a, b])
        sim = float(vectors[0] @ vectors[1])
        verdict = ("DUPLICATE" if sim >= config.DUPLICATE_SIM_THRESHOLD
                   else "partial" if sim > config.DIVERSE_SIM_THRESHOLD
                   else "diverse")
        print(f"         {a!r} vs {b!r}: {sim:.3f}  ({verdict})")

    sampler_ok = bool(config.LLM_API_KEY)
    print(f"\nLIVE: LLM ensemble configured = {sampler_ok}")
    if sampler_ok:
        tagger = EdgeTagger()
        for post in POSTS[:2]:
            started = time.perf_counter()
            tags = tagger.generate_tags(post, timeout=10.0)
            print(f"       {time.perf_counter() - started:5.2f}s  {tags}")
            check("live run returned tags", bool(tags))


def main() -> int:
    print("=" * 70)
    print("sn101_edge self-test")
    print("=" * 70)
    test_format_gate()
    test_stemmer_symmetry()
    test_verbatim_guarantee()
    test_ngrams()
    test_repair()
    test_parser()
    test_selection()
    test_fallback()
    test_deadline()

    if "--live" in sys.argv:
        test_live()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
