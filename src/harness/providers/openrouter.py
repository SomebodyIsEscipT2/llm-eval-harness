"""OpenRouter provider.

OpenRouter model IDs use the ``<provider>/<model>`` form, e.g.
``openai/gpt-4o-mini`` or ``anthropic/claude-3.5-sonnet``.

The ``openrouter/`` prefix used elsewhere in some harnesses is a routing
convention, not part of OpenRouter's own API; if a caller passes
``openrouter/openai/gpt-4o-mini`` we strip the leading ``openrouter/`` so
the request still works.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from harness.providers import register
from harness.providers.base import Completion, Provider, ProviderError, Usage

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def normalize_model(model: str) -> str:
    """Strip a leading ``openrouter/`` routing prefix if present."""
    prefix = "openrouter/"
    return model[len(prefix):] if model.startswith(prefix) else model


@register("openrouter")
class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
        app_name: str = "llm-eval-harness",
        verify: Any = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set and no api_key was provided."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.app_name = app_name
        # TLS verification: explicit value wins; otherwise honour
        # HARNESS_TLS_VERIFY (e.g. behind a TLS-intercepting egress proxy).
        if verify is not None:
            self.verify = verify
        else:
            env = os.environ.get("HARNESS_TLS_VERIFY")
            if env is None:
                self.verify = True
            elif env.strip().lower() in ("0", "false", "no", "off"):
                self.verify = False
            else:
                self.verify = env  # path to a CA bundle
        self._client = client
        self._owns_client = client is None

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
            "model": normalize_model(model or self.model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }

        client = self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:  # network-level failure
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenRouter returned {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected OpenRouter response shape: {data}") from exc

        return Completion(
            text=text,
            model=data.get("model", payload["model"]),
            usage=_usage_from(data),
            latency_ms=latency_ms,
            raw=data,
        )
    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _usage_from(data: Dict[str, Any]) -> Usage:
    u = data.get("usage") or {}
    return Usage(
        prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
        completion_tokens=int(u.get("completion_tokens", 0) or 0),
        total_tokens=int(u.get("total_tokens", 0) or 0),
    )
