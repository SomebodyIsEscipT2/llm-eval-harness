"""Provider registry."""

from __future__ import annotations

from typing import Dict, Type

from harness.providers.base import Provider

_REGISTRY: Dict[str, Type[Provider]] = {}


def register(name: str):
    """Decorator: register a provider class under ``name``."""

    def _wrap(cls: Type[Provider]) -> Type[Provider]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def available() -> Dict[str, Type[Provider]]:
    return dict(_REGISTRY)


def get(name: str) -> Type[Provider]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown provider {name!r}. Available: {sorted(_REGISTRY) or 'none'}"
        )
    return _REGISTRY[name]


# Import built-in providers so they self-register.
from harness.providers import openrouter as _openrouter  # noqa: E402,F401
from harness.providers import openai_compat as _openai  # noqa: E402,F401
from harness.providers import ollama as _ollama  # noqa: E402,F401
