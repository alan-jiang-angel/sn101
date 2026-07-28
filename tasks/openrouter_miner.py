"""OpenRouter-powered SN101 miner for tag101.

This module is designed to be run by the tag101 miner runtime with
"--task.miner_module tag101.tasks.openrouter_miner" or loaded directly by
custom scripts.
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np

from .framework.base import TaskHandler
from .sn101 import KIND, SPEC_VERSION, score_answers
from .sn101_reference.core.scoring.preprocessing import normalize_tag
from ..protocol import TaskEnvelope

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
DEFAULT_N_TAGS = 3
CANDIDATE_COUNT = 8
REQUEST_TIMEOUT = 45

TAGGING_SYSTEM_PROMPT = """\
You are a tag generation assistant.
Generate concise topic tags for a social post.
Return only a JSON array of distinct lower-case strings.
Do not include hashtags, URLs, punctuation-only strings, or extra explanation.
Each tag should use 1-5 words and be semantically meaningful.
Prioritize tags that are specific to the post content and its main entities.
"""

TAGGING_USER_PROMPT = """\
Post:
{post}

Return exactly {candidate_count} tags if possible.
Choose tags that are highly relevant to the main topic, entities, and context of the post.
Avoid generic buzzwords and near-duplicate tags.
Output only a JSON array.
"""

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(com|org|net|io|ai|co)\b)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class OpenRouterMiner:
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = OPENROUTER_MODEL,
        n_tags: int = DEFAULT_N_TAGS,
        timeout_sec: int = REQUEST_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.n_tags = n_tags
        self.timeout_sec = timeout_sec

    def generate_tags(self, post: str) -> list[str]:
        if not post or not post.strip():
            return []
        if not self.api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY for OpenRouter tag generation."
            )

        content = self._call_chat_completions(post)
        candidates = self._parse_tags(content)
        candidates = self._normalize_and_filter_candidates(candidates)

        if not candidates:
            return self._fallback_tags(post)

        selected = self._select_top_tags(post, candidates)
        if len(selected) < self.n_tags:
            selected = self._fill_tags(post, selected, candidates)

        return selected[: self.n_tags]

    def _call_chat_completions(self, post: str) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TAGGING_SYSTEM_PROMPT},
                {"role": "user", "content": TAGGING_USER_PROMPT.format(post=post.strip(), candidate_count=CANDIDATE_COUNT)},
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }
        request = Request(
            url=endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_text = response.read().decode("utf-8")
            data = json.loads(response_text)
            return data["choices"][0]["message"]["content"]
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    def _parse_tags(self, raw_content: str) -> list[str]:
        text = raw_content.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            ).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = re.split(r"[\n,;|]+", text)

        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if isinstance(item, str)]

    def _normalize_and_filter_candidates(self, candidates: list[str]) -> list[str]:
        seen: set[str] = set()
        filtered: list[str] = []
        for candidate in candidates:
            tag = normalize_tag(candidate)
            if not tag or tag in seen:
                continue
            if self._is_valid_tag(tag):
                seen.add(tag)
                filtered.append(tag)
        return filtered

    def _is_valid_tag(self, tag: str) -> bool:
        if not tag:
            return False
        if URL_PATTERN.search(tag):
            return False
        if tag.count(" ") >= 5:
            return False

        tokens = self._tokenize(tag)
        if not tokens or len(tokens) > 5:
            return False
        if self._is_numeric_only(tag):
            return False
        if self._is_punctuation_only(tag):
            return False
        if self._is_emoji_only(tag):
            return False
        if self._has_too_many_special_symbols(tag):
            return False
        if any(token.startswith("#") or token.startswith("@") for token in tokens):
            return False
        return True

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def _is_numeric_only(self, text: str) -> bool:
        return bool(re.fullmatch(r"[\d\s.,:/%+\-]+", text))

    def _is_punctuation_only(self, text: str) -> bool:
        return bool(re.fullmatch(r"[^\w\s]+", text, flags=re.UNICODE))

    def _is_emoji_only(self, text: str) -> bool:
        compact = "".join(ch for ch in text if not ch.isspace())
        if not compact:
            return False
        if any(ch.isalnum() for ch in compact):
            return False
        return all(
            unicodedata.category(ch).startswith(("S", "P"))
            for ch in compact
        )

    def _has_too_many_special_symbols(self, text: str) -> bool:
        compact = [ch for ch in text if not ch.isspace()]
        if not compact:
            return False
        special_count = sum(1 for ch in compact if not ch.isalnum() and ch not in {"-", "_", "&", "/", "+"})
        return (special_count / len(compact)) > 0.3

    def _select_top_tags(self, post: str, tags: list[str]) -> list[str]:
        if len(tags) <= self.n_tags:
            return tags[: self.n_tags]

        post_tokens = set(self._tokenize(post))
        embedding_scores = self._embedding_relevance(post=post, tags=tags)
        candidates: list[tuple[str, float]] = []
        for tag in tags:
            lexical_relevance = self._lexical_relevance(tag=tag, post_tokens=post_tokens)
            if embedding_scores:
                embedding_score = embedding_scores[tags.index(tag)]
                relevance = (embedding_score * 0.6) + (lexical_relevance * 0.4)
            else:
                relevance = lexical_relevance
            candidates.append((tag, relevance))

        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        selected: list[str] = []
        for tag, relevance in ordered:
            if len(selected) >= self.n_tags:
                break
            if not selected:
                selected.append(tag)
                continue

            diversity_bonus = self._diversity_bonus(tag=tag, selected=selected)
            combined_score = (relevance * 0.7) + (diversity_bonus * 0.3)
            if combined_score >= 0.2:
                selected.append(tag)

        if len(selected) < self.n_tags:
            selected.extend(tag for tag, _ in ordered if tag not in selected)
        return selected[: self.n_tags]

    def _embedding_relevance(self, post: str, tags: list[str]) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return []

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return []

        text_embeddings = model.encode(
            [post] + tags,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if np.ndim(text_embeddings) == 1:
            text_embeddings = np.array([text_embeddings])
        post_embedding = text_embeddings[0]
        tag_embeddings = text_embeddings[1:]
        return [
            float(max(0.0, min(1.0, float(np.dot(tag_embedding, post_embedding)))))
            for tag_embedding in tag_embeddings
        ]

    def _lexical_relevance(self, tag: str, post_tokens: set[str]) -> float:
        tag_tokens = set(self._tokenize(tag))
        if not tag_tokens:
            return 0.0
        overlap = len(tag_tokens & post_tokens) / max(1, len(tag_tokens))
        if normalize_tag(tag) and normalize_tag(tag) in normalize_tag(" ".join(post_tokens)):
            overlap = max(overlap, 0.9)
        return min(1.0, overlap)

    def _diversity_bonus(self, tag: str, selected: list[str]) -> float:
        if not selected:
            return 1.0
        tag_tokens = set(self._tokenize(tag))
        selected_tokens = {token for item in selected for token in self._tokenize(item)}
        if not tag_tokens:
            return 0.0
        overlap = len(tag_tokens & selected_tokens) / max(1, len(tag_tokens))
        return max(0.0, 1.0 - overlap)

    def _fill_tags(self, post: str, selected: list[str], candidates: list[str]) -> list[str]:
        for candidate in candidates:
            if candidate in selected:
                continue
            if len(selected) >= self.n_tags:
                break
            selected.append(candidate)
        if len(selected) < self.n_tags:
            selected.extend(self._fallback_tags(post))
        return selected[: self.n_tags]

    def _fallback_tags(self, post: str) -> list[str]:
        tokens = [token for token in self._tokenize(post) if len(token) > 2]
        default_tags = []
        for token in tokens:
            tag = normalize_tag(token)
            if tag and self._is_valid_tag(tag):
                default_tags.append(tag)
            if len(default_tags) >= self.n_tags:
                break

        generic_tags = [
            "ai",
            "artificial intelligence",
            "machine learning",
            "social media",
            "technology",
            "natural language processing",
        ]
        for tag in generic_tags:
            if self._is_valid_tag(tag) and tag not in default_tags:
                default_tags.append(tag)
            if len(default_tags) >= self.n_tags:
                break
        return default_tags[: self.n_tags]


def solve_problem(envelope: TaskEnvelope, chain_runtime: Any) -> dict[str, Any]:
    post = str(envelope.payload.get("text", ""))
    print(post)
    miner = OpenRouterMiner()
    try:
        tags = miner.generate_tags(post)
    except RuntimeError:
        tags = random.sample(
            ["bitcoin", "ethereum", "crypto market", "defi", "stablecoin", "regulation", "etf", "trading"],
            k=DEFAULT_N_TAGS,
        )
    return {"tags": tags}


def handler() -> TaskHandler:
    return TaskHandler(
        kind=KIND,
        spec_version=SPEC_VERSION,
        solve_problem=solve_problem,
        score_answers=score_answers,
        description="OpenRouter-powered SN101 miner task handler.",
    )
