"""LLM provider abstraction.

The API doesn't hardcode a single model provider. Instead, `get_provider()`
returns one of:

- AnthropicProvider  -> used automatically if ANTHROPIC_API_KEY is set,
                        streams real completions from Claude.
- MockProvider       -> used otherwise (and always in tests), so the whole
                        project can be run, demoed, and tested with zero
                        API keys or network access.

Both providers expose the same interface: an async generator that yields
text chunks, so main.py doesn't need to know which one it's talking to.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Protocol


class LLMProvider(Protocol):
    # Note: plain `def` (not `async def`) here on purpose. An async
    # generator *function* like `async def f(): yield x` is called like a
    # regular function and returns an AsyncIterator directly (no `await`
    # needed on the call itself) — so that's the signature this Protocol
    # describes, matching how MockProvider/AnthropicProvider are written
    # below and how main.py calls them (`async for chunk in provider...`).
    def stream_reply(
        self, system_prompt: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Yield the assistant's reply piece by piece."""
        ...


class MockProvider:
    """Deterministic fake provider — no network, no API key, no cost.

    It "streams" back a canned reply word by word so the SSE endpoint has
    real chunks to send. Good enough to prove the streaming plumbing works
    end to end, which is what the test suite checks.
    """

    async def stream_reply(
        self, system_prompt: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        last_user_message = history[-1]["content"] if history else ""
        reply = f"(mock reply, system_prompt={system_prompt!r}) You said: {last_user_message}"
        for word in reply.split(" "):
            await asyncio.sleep(0)  # yield control, like a real streaming call would
            yield word + " "


class AnthropicProvider:
    """Streams real completions from the Anthropic API.

    Only constructed when ANTHROPIC_API_KEY is present in the environment.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model

    async def stream_reply(
        self, system_prompt: str, history: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        # Imported lazily so the `anthropic` package is only required if
        # you actually configure a real API key.
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self._api_key)
        async with client.messages.stream(
            model=self._model,
            max_tokens=1000,
            system=system_prompt,
            messages=history,
        ) as stream:
            async for text in stream.text_stream:
                yield text


def get_provider() -> LLMProvider:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return AnthropicProvider(api_key=api_key)
    return MockProvider()
