"""OpenAI-compatible provider.

Works with any endpoint that speaks the OpenAI chat-completions shape:
OpenAI itself, Groq, Together, vLLM, LM Studio, and local servers. Set the
base URL to point elsewhere.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from harness.providers import register
from harness.providers.base import Completion, Provider, ProviderError, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@register("openai_compat")
class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
        verify: Any = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderError(
                "OPENAI_API_KEY is not set and no api_key was provided."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self.verify = verify if verify is not None else True

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, verify=self.verify)
            self._owns_client = True
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code >= 400:
            raise ProviderError(f"returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {data}") from exc

        u = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", payload["model"]),
            usage=Usage(
                int(u.get("prompt_tokens", 0) or 0),
                int(u.get("completion_tokens", 0) or 0),
                int(u.get("total_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            raw=data,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
