from __future__ import annotations

import json
import os
import re
import typing
from urllib.request import Request, urlopen

# Players can paste a key here for a quick local run, or set OPENAI_API_KEY.
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "your openai key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_N_TAGS = 3

TAGGING_SYSTEM_PROMPT = """\
Generate topic tags for a social post.
Return only a JSON array of lower-case strings.
Output exactly {n_tags} tags.
"""

TAGGING_USER_PROMPT = """\
Create tags for the post. Return a JSON array only, with no duplicates or extra text.
Post:
{post}
"""


class Miner:
    """
    Example miner for players.

    Replace generate_tags with your own logic. The only expected interface is:
    input a post string, return a list of tags.
    """

    def __init__(
        self,
        api_key: str = OPENAI_KEY,
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_MODEL,
        n_tags: int = DEFAULT_N_TAGS,
        timeout_sec: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.n_tags = n_tags
        self.timeout_sec = timeout_sec

    def generate_tags(
        self,
        post: str,
    ) -> list[str]:
        if not post.strip():
            return []
        if not self.api_key or self.api_key == "your openai key":
            raise RuntimeError(
                "Set OPENAI_API_KEY or replace OPENAI_KEY in core/miner.py."
            )

        system_prompt = TAGGING_SYSTEM_PROMPT.format(n_tags=self.n_tags)
        user_prompt = TAGGING_USER_PROMPT.format(
            post=post.strip(),
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        content = self._call_chat_completions(payload)
        return self._parse_tags(content)

    def _call_chat_completions(self, payload: dict[str, typing.Any]) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
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
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    def _parse_tags(self, raw_content: str) -> list[str]:
        text = raw_content.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            ).strip()

        try:
            tags = json.loads(text)
        except json.JSONDecodeError:
            tags = re.split(r"[,\n]", text)

        if not isinstance(tags, list):
            return []
        return tags
