"""Ollama provider for fully local models.

No API key: talks to a local Ollama server (default http://localhost:11434).
Ollama reports token counts as ``prompt_eval_count`` and ``eval_count``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from harness.providers import register
from harness.providers.base import Completion, Provider, ProviderError, Usage

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"


@register("ollama")
class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
        verify: Any = None,
    ) -> None:
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
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        payload.update(kwargs)

        client = self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        text = data.get("response", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return Completion(
            text=text,
            model=data.get("model", payload["model"]),
            usage=Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            latency_ms=latency_ms,
            raw=data,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
