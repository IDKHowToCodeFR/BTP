"""LLM backend abstraction.

The agent's reasoning module (reasoning.py) only ever calls
``backend.complete(system_prompt, user_prompt) -> str``; it does not know
or care which provider is behind that call. Four backends are provided:

- AnthropicBackend: budget-tier Claude model by default (see MODEL
  NOTE below) via the official `anthropic` package.
- OpenAIBackend: any OpenAI-compatible chat model via the `openai`
  package (also works against many self-hosted OpenAI-compatible
  servers by overriding `base_url`).
- OllamaBackend: a locally-running model (e.g. Llama 3.1 8B) via
  Ollama's local HTTP API, for zero per-call API cost.
- MockBackend: fully deterministic, no network call, no API key. Used
  in this project's own test suite, and useful for wiring-up an
  end-to-end dry run before spending any real API budget.

MODEL NOTE: the default Anthropic model string below
("claude-haiku-4-5-20251001") is the current budget/low-latency tier
Claude model as of this project being built. Model names change over
time -- if this string 404s for you, check
https://docs.claude.com for the current model list and pass a different
`model=` argument; nothing else in this codebase needs to change.
"""
from __future__ import annotations

import abc
import os
from typing import Callable, List, Optional, Sequence, Tuple


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        """Return the raw text of the model's response and a dictionary of token usage."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class AnthropicBackend(LLMBackend):
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package: "
                "pip install anthropic"
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key found. Pass api_key=... or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        usage = {
            "prompt_tokens": resp.usage.input_tokens if hasattr(resp, "usage") else 0,
            "completion_tokens": resp.usage.output_tokens if hasattr(resp, "usage") else 0,
        }
        return text, usage


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires the 'openai' package: pip install openai"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Pass api_key=... or set the "
                "OPENAI_API_KEY environment variable."
            )
        self._client = openai.OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if hasattr(resp, "usage") and resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if hasattr(resp, "usage") and resp.usage else 0,
        }
        return text, usage


class OllamaBackend(LLMBackend):
    """Zero-API-cost local backend, e.g. `ollama pull llama3.1:8b` then
    `ollama serve` (usually already running as a background service after
    install). No API key needed.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 300,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        import requests
        import time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                
                # Ollama prompt/completion eval count
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
                return text, usage
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Ollama backend failed after {max_retries} attempts: {e}") from e
                
                # Exponential backoff: 3s, 6s, 12s, 24s...
                time.sleep(3 * (2 ** attempt))


class MockBackend(LLMBackend):
    """No network call, no API key, fully deterministic. Three modes:

    1. `fixed_response` -- always return this exact string.
    2. `responses` -- cycle through this list, one per call (repeats the
       last entry once exhausted). Useful for scripting a specific
       sequence, e.g. testing that the Nth call triggers a fallback.
    3. `responder` -- an arbitrary callable(system_prompt, user_prompt)
       -> str, for fully custom test logic (e.g. echoing back a
       lookup-table weight vector for whatever task profile keyword it
       finds in user_prompt, to smoke-test the full pipeline without
       ever calling a real model).

    Exactly one of the three should be provided.
    """

    def __init__(
        self,
        fixed_response: Optional[str] = None,
        responses: Optional[Sequence[str]] = None,
        responder: Optional[Callable[[str, str], str]] = None,
    ):
        modes_given = sum(x is not None for x in (fixed_response, responses, responder))
        if modes_given != 1:
            raise ValueError(
                "MockBackend requires exactly one of fixed_response, "
                "responses, or responder"
            )
        self._fixed_response = fixed_response
        self._responses: List[str] = list(responses) if responses is not None else []
        self._responder = responder
        self._call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        self._call_count += 1
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        if self._responder is not None:
            return self._responder(system_prompt, user_prompt), usage
        if self._fixed_response is not None:
            return self._fixed_response, usage
        idx = min(self._call_count - 1, len(self._responses) - 1)
        return self._responses[idx], usage


def build_backend_from_config(config: dict) -> LLMBackend:
    """Construct a backend from a plain dict, e.g. loaded from config.yaml:

        llm:
          provider: anthropic   # anthropic | openai | ollama | mock
          model: claude-haiku-4-5-20251001
          temperature: 0.0
          max_tokens: 1000

    `mock` provider in a config file only supports fixed_response (for
    a reproducible dry-run smoke test), since responses/responder are
    Python callables/lists that don't serialize to YAML.
    """
    provider = config.get("provider", "mock").lower()
    if provider == "anthropic":
        return AnthropicBackend(
            model=config.get("model", "claude-haiku-4-5-20251001"),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.0),
        )
    if provider == "openai":
        return OpenAIBackend(
            model=config.get("model", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            max_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.0),
        )
    if provider == "ollama":
        return OllamaBackend(
            model=config.get("model", "llama3.1:8b"),
            host=config.get("host", "http://localhost:11434"),
            temperature=config.get("temperature", 0.0),
        )
    if provider == "mock":
        return MockBackend(fixed_response=config.get("fixed_response", "{}"))
    raise ValueError(f"unknown llm provider: {provider!r}")
