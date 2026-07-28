"""Estimate what the rest of the subnet will submit.

Consensus is 60% of the reward and it is a pure popularity contest: a tag's
score is (fraction of miners in its cluster) x exp(-(centroid_rank - 1)). The
rank term is savage -- being the second-closest surface form to a cluster
centroid costs you a factor of e. Since the centroid is a frequency-weighted
mean, the MODAL surface form wins rank 1 almost always.

So the job is not "produce good tags", it is "predict the tag strings the
plurality of miners will produce". The single best predictor available is the
reference miner's own prompt run against the model the reference miner
defaults to, because that is literally what most of the field is running. We
sample it repeatedly at temperature to recover its output distribution, and mix
in other models for coverage of the same semantic space.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from . import config
from .constraints import PostContext, dedupe_preserving_order, repair_tag

# Reproduced from tag101/tasks/sn101_reference/core/miner.py so our samples come
# from the same conditional distribution as the crowd's.
REFERENCE_SYSTEM_PROMPT = """\
Generate topic tags for a social post.
Return only a JSON array of lower-case strings.
Output exactly {n_tags} tags.
"""

REFERENCE_USER_PROMPT = """\
Create tags for the post. Return a JSON array only, with no duplicates or extra text.
Post:
{post}
"""

# A second prompt that pushes toward literal spans of the post. Tags drawn from
# post tokens are guaranteed validity 1.0, so these candidates are strictly
# safer when they are also popular.
EXTRACTIVE_SYSTEM_PROMPT = """\
You extract topic tags from social posts.
Rules:
- Return ONLY a JSON array of lower-case strings.
- Output exactly {n_tags} tags.
- Every tag MUST use only words that literally appear in the post.
- Each tag is 1 to 3 words. No hashtags, no URLs, no punctuation, no numbers alone.
- Prefer the single most obvious, conventional name for each main topic.
- The three tags must cover three DIFFERENT aspects of the post.
"""

EXTRACTIVE_USER_PROMPT = """\
Post:
{post}

Return the JSON array only.
"""


@dataclass
class CrowdEstimate:
    """Candidate tags with an estimated popularity mass."""

    counts: Counter = field(default_factory=Counter)
    samples: list[list[str]] = field(default_factory=list)
    n_samples: int = 0
    errors: list[str] = field(default_factory=list)

    def frequency(self, tag: str) -> float:
        if self.n_samples <= 0:
            return 0.0
        return self.counts.get(tag, 0) / self.n_samples

    def ranked(self, limit: int | None = None) -> list[tuple[str, float]]:
        items = [(t, self.frequency(t)) for t, _ in self.counts.most_common()]
        return items[:limit] if limit else items

    def as_simulated_miners(self, max_miners: int = 40) -> list[list[str]]:
        """The sampled tag sets, reusable as a synthetic crowd for scoring."""
        return [s for s in self.samples if s][:max_miners]


def _parse_tags(raw: str) -> list[str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    # Prefer a JSON array anywhere in the response.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [t for t in parsed if isinstance(t, str)]
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [t for t in parsed if isinstance(t, str)]
    except json.JSONDecodeError:
        pass
    return [p for p in re.split(r"[,\n]", text) if p.strip()]


class CrowdSampler:
    """Fires the ensemble concurrently under a hard deadline."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
    ):
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.models = models or config.ENSEMBLE_MODELS

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and self.api_key != "your openai key"

    async def sample(self, post: str, ctx: PostContext, budget: float) -> CrowdEstimate:
        estimate = CrowdEstimate()
        if not self.enabled or budget <= 0:
            estimate.errors.append("llm disabled or no budget")
            return estimate

        requests = self._build_requests(post)
        semaphore = asyncio.Semaphore(config.LLM_MAX_CONCURRENCY)

        try:
            import httpx
        except ImportError:
            estimate.errors.append("httpx not installed")
            return estimate

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in self.base_url:
            headers["X-Title"] = "sn101-edge"

        async with httpx.AsyncClient(timeout=budget, headers=headers) as client:
            async def one(body: dict) -> list[str] | None:
                async with semaphore:
                    try:
                        resp = await client.post(
                            f"{self.base_url}/chat/completions", json=body
                        )
                        if resp.status_code != 200:
                            estimate.errors.append(
                                f"{body['model']}:{resp.status_code}"
                            )
                            return None
                        data = resp.json()
                        return _parse_tags(
                            data["choices"][0]["message"]["content"]
                        )
                    except Exception as exc:  # noqa: BLE001 - never fail a task
                        estimate.errors.append(f"{body.get('model')}:{type(exc).__name__}")
                        return None

            tasks = [asyncio.create_task(one(body)) for body in requests]
            done, pending = await asyncio.wait(tasks, timeout=budget)
            for task in pending:
                task.cancel()
            results = []
            for task in done:
                try:
                    results.append(task.result())
                except Exception:  # noqa: BLE001
                    results.append(None)

        for raw_tags in results:
            if not raw_tags:
                continue
            repaired = [repair_tag(t, ctx) for t in raw_tags]
            cleaned = dedupe_preserving_order([t for t in repaired if t])
            if not cleaned:
                continue
            estimate.samples.append(cleaned)
            estimate.n_samples += 1
            for tag in cleaned:
                estimate.counts[tag] += 1

        return estimate

    def _build_requests(self, post: str) -> list[dict]:
        bodies: list[dict] = []
        n = config.N_TAGS
        for model in self.models:
            for i in range(config.ENSEMBLE_SAMPLES_PER_MODEL):
                # Alternate prompts: the reference prompt reproduces the crowd,
                # the extractive prompt yields guaranteed-validity candidates.
                if i % 2 == 0:
                    system = REFERENCE_SYSTEM_PROMPT.format(n_tags=n)
                    user = REFERENCE_USER_PROMPT.format(post=post.strip())
                else:
                    system = EXTRACTIVE_SYSTEM_PROMPT.format(n_tags=n)
                    user = EXTRACTIVE_USER_PROMPT.format(post=post.strip())
                bodies.append(
                    {
                        "model": model,
                        "temperature": config.ENSEMBLE_TEMPERATURE,
                        "max_tokens": 120,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    }
                )
        return bodies
