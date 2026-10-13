"""Tests for the OpenRouter provider using a mocked HTTP transport."""

import httpx
import pytest
import respx

from harness.providers.base import ProviderError
from harness.providers.openrouter import OpenRouterProvider


@pytest.fixture
def provider():
    client = httpx.AsyncClient()
    p = OpenRouterProvider(api_key="test-key", client=client)
    yield p


@respx.mock
async def test_complete_success(provider):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "openrouter/openai/gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "4"}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 1,
                    "total_tokens": 13,
                },
            },
        )
    )
    completion = await provider.complete("What is 2+2?", model="openrouter/openai/gpt-4o-mini")
    assert completion.text == "4"
    assert completion.usage.total_tokens == 13
    assert completion.usage.prompt_tokens == 12
    assert completion.latency_ms >= 0


@respx.mock
async def test_complete_http_error(provider):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    with pytest.raises(ProviderError) as exc:
        await provider.complete("hi")
    assert "401" in str(exc.value)


@respx.mock
async def test_complete_network_error(provider):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("boom")
    )
    with pytest.raises(ProviderError):
        await provider.complete("hi")


@respx.mock
async def test_complete_malformed_response(provider):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": True})
    )
    with pytest.raises(ProviderError):
        await provider.complete("hi")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenRouterProvider(api_key=None)


async def test_aclose_is_safe(provider):
    await provider.aclose()
    await provider.aclose()  # idempotent
