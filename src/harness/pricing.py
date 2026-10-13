"""Model price table for cost estimation.

Prices are USD per 1,000,000 tokens, keyed by OpenRouter-style model ID
(``<provider>/<model>``). They are **estimates** — providers change prices,
and this table is deliberately small. A missing model yields no cost rather
than a wrong one: we never invent a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from harness.providers.base import Usage


@dataclass(frozen=True)
class Price:
    prompt_per_m: float
    completion_per_m: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.prompt_tokens / 1_000_000 * self.prompt_per_m
            + usage.completion_tokens / 1_000_000 * self.completion_per_m
        )


# USD per 1M tokens. Update as providers change their pricing.
DEFAULT_PRICES: Dict[str, Price] = {
    "openai/gpt-4o-mini": Price(0.15, 0.60),
    "openai/gpt-4o": Price(2.50, 10.00),
    "anthropic/claude-3.5-sonnet": Price(3.00, 15.00),
    "anthropic/claude-3-haiku": Price(0.25, 1.25),
    "google/gemini-1.5-flash": Price(0.075, 0.30),
    "meta-llama/llama-3.1-8b-instruct": Price(0.06, 0.06),
}


def load_prices(path: Optional[str] = None) -> Dict[str, Price]:
    """Built-in table, optionally overlaid with a JSON file of the same shape."""
    prices = dict(DEFAULT_PRICES)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Price file not found: {p}")
        overlay = json.loads(p.read_text(encoding="utf-8"))
        for model, spec in overlay.items():
            prices[model] = Price(
                float(spec["prompt_per_m"]), float(spec["completion_per_m"])
            )
    return prices


def estimate(model: str, usage: Usage, prices: Optional[Dict[str, Price]] = None) -> Optional[float]:
    """Estimated USD for a usage record, or None if the model isn't priced."""
    table = prices if prices is not None else DEFAULT_PRICES
    price = table.get(model)
    if price is None:
        return None
    return price.cost(usage)
