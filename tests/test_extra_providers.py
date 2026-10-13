"""Tests for the OpenAI-compatible and Ollama providers (mocked)."""

import httpx
import pytest
import respx

from harness.providers.base import ProviderError
from harness.providers.ollama import OllamaProvider
from harness.providers.openai_compat import OpenAICompatProvider


@pytest.fixture
def openai_provider():
    return OpenAICompatProvider(api_key="***", client=httpx.AsyncClient())


@respx.mock
async def test_openai_compat_success(openai_provider):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "4"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )
    )
    c = await openai_provider.complete("2+2", model="gpt-4o-mini")
    assert c.text == "4"
    assert c.usage.total_tokens == 6


@respx.mock
async def test_openai_compat_http_error(openai_provider):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    with pytest.raises(ProviderError) as exc:
        await openai_provider.complete("hi")
    assert "429" in str(exc.value)


@respx.mock
async def test_openai_compat_custom_base_url():
    p = OpenAICompatProvider(
        api_key="***", base_url="http://localhost:8000/v1", client=httpx.AsyncClient()
    )
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"model": "local", "choices": [{"message": {"content": "ok"}}], "usage": {}},
        )
    )
    c = await p.complete("hi")
    assert c.text == "ok"


def test_openai_compat_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatProvider(api_key=None)


@pytest.fixture
def ollama_provider():
    return OllamaProvider(client=httpx.AsyncClient())


@respx.mock
async def test_ollama_success(ollama_provider):
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "response": "4",
                "prompt_eval_count": 12,
                "eval_count": 1,
            },
        )
    )
    c = await ollama_provider.complete("2+2")
    assert c.text == "4"
    assert c.usage.prompt_tokens == 12
    assert c.usage.completion_tokens == 1
    assert c.usage.total_tokens == 13


@respx.mock
async def test_ollama_error(ollama_provider):
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(500, text="model not found")
    )
    with pytest.raises(ProviderError) as exc:
        await ollama_provider.complete("hi")
    assert "500" in str(exc.value)


@respx.mock
async def test_ollama_network_error(ollama_provider):
    respx.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(ProviderError):
        await ollama_provider.complete("hi")


def test_registry_has_all_providers():
    from harness.providers import available

    names = set(available())
    assert {"openrouter", "openai_compat", "ollama"} <= names
