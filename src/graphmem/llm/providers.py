
"""Mock and OpenAI-compatible LLM providers."""

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from graphmem.llm.base import LLMProvider, LLMResponse, Message


class MockProvider(LLMProvider):
    """Return scripted responses without making network requests."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[list[Message]] = []
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def generate(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._index >= len(self.script):
            raise RuntimeError("MockProvider script exhausted")

        prompt = list(messages)
        self.prompts.append(prompt)
        self.calls.append(
            {
                "messages": prompt,
                "tools": tools,
            }
        )

        text = self.script[self._index]
        self._index += 1

        return LLMResponse(
            text=text,
            tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            latency_ms=0.0,
        )


class OpenAICompatibleProvider(LLMProvider):
    """Call an OpenAI-compatible chat completions endpoint.

    Configuration can be passed explicitly or through:
      GRAPHMEM_LLM_BASE_URL
      GRAPHMEM_LLM_MODEL
      GRAPHMEM_LLM_API_KEY
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 1000.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv(
                "GRAPHMEM_LLM_BASE_URL",
                "http://localhost:11434/v1",
            )
        ).rstrip("/")
        self.model = model or os.getenv(
            "GRAPHMEM_LLM_MODEL",
            "llama3.2",
        )
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("GRAPHMEM_LLM_API_KEY")
        )
        self.timeout = timeout

    def generate(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }

        if tools is not None:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        started = time.perf_counter()

        # Two attempts total: the original request and one retry.
        for attempt in range(2):
            try:
                response = httpx.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Retry transient server failures and rate limiting once.
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == 0:
                        time.sleep(0.2)
                        continue

                response.raise_for_status()
                data = response.json()
                break

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 1:
                    raise
                time.sleep(0.2)

            except httpx.HTTPStatusError as exc:
                if attempt == 0 and exc.response.status_code in (
                    429, 500, 502, 503, 504
                ):
                    time.sleep(0.2)
                    continue
                raise
        else:
            raise RuntimeError("LLM request failed after retry")

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("LLM response contains no choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []

        raw_usage = data.get("usage") or {}

        return LLMResponse(
            text=content if isinstance(content, str) else "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(
                    raw_usage.get("completion_tokens") or 0
                ),
            },
            latency_ms=elapsed_ms,
        )
