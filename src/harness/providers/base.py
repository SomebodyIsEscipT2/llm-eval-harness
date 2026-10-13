"""Provider abstraction: one interface, many LLM backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Usage:
    """Token usage for a single call, when the provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class Completion:
    """A single provider response plus the metadata the harness records."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when a provider call fails irrecoverably."""


class Provider(abc.ABC):
    """Base class every backend implements.

    A provider is deliberately small: given a prompt (and optional system
    message), return a ``Completion``. Everything else — scoring, retries
    across cases, reporting — belongs to the harness, not the provider.
    """

    name: str = "provider"

    @abc.abstractmethod
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
        """Return a completion for ``prompt``.

        Implementations must be side-effect free with respect to the
        harness: no retries, no scoring, no persistence.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any network resources. Safe to call more than once."""
        return None

    @property
    def models(self) -> List[str]:
        """Models this provider is known to expose. May be empty."""
        return []
