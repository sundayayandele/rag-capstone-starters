"""Optional hosted LLM providers (stdlib only, no SDK needed).

RAG_LLM = none (default, offline extractive baseline) | github | anthropic
  github:    GitHub Models, uses GITHUB_TOKEN (needs `models: read` permission in Actions). RAG_LLM_MODEL default openai/gpt-4.1-mini
  anthropic: uses ANTHROPIC_API_KEY. RAG_LLM_MODEL default claude-haiku-4-5

Provider endpoints and model names change; check the provider docs if a call fails.
"""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict, retries: int = 3, timeout: int = 60) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(str(e)) from e
    raise LLMError("unreachable")


class GitHubModels:
    name = "github"

    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not self.token:
            raise LLMError("GITHUB_TOKEN is not set")
        self.model = os.getenv("RAG_LLM_MODEL", "openai/gpt-4.1-mini")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = _post("https://models.github.ai/inference/chat/completions",
                  {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
                  {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return r["choices"][0]["message"]["content"].strip()


class Anthropic:
    name = "anthropic"

    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.model = os.getenv("RAG_LLM_MODEL", "claude-haiku-4-5")

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                  {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                   "messages": [{"role": "user", "content": user}]})
        return "".join(b.get("text", "") for b in r["content"]).strip()


def get_llm(name: str | None = None):
    """Return a provider, or None for the offline extractive baseline."""
    name = (name or os.getenv("RAG_LLM", "none")).lower()
    if name == "github":
        return GitHubModels()
    if name == "anthropic":
        return Anthropic()
    return None
