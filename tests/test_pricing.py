"""Tests for cost estimation."""

import pytest

from harness.pricing import Price, estimate, load_prices
from harness.providers.base import Usage


def test_price_cost_calculation():
    p = Price(0.15, 0.60)  # gpt-4o-mini
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert p.cost(usage) == pytest.approx(0.75)


def test_price_small_usage():
    p = Price(1.0, 2.0)
    usage = Usage(prompt_tokens=1000, completion_tokens=500)
    # 1000/1e6 * 1.0 + 500/1e6 * 2.0 = 0.001 + 0.001
    assert p.cost(usage) == pytest.approx(0.002)


def test_estimate_known_model():
    usage = Usage(1_000_000, 0, 1_000_000)
    cost = estimate("openai/gpt-4o-mini", usage)
    assert cost == pytest.approx(0.15)


def test_estimate_unknown_model_returns_none():
    assert estimate("mystery/model", Usage(100, 100, 200)) is None


def test_load_prices_overlay(tmp_path):
    p = tmp_path / "prices.json"
    p.write_text('{"my/model": {"prompt_per_m": 1.0, "completion_per_m": 2.0}}')
    prices = load_prices(str(p))
    assert "my/model" in prices
    assert "openai/gpt-4o-mini" in prices  # built-ins preserved


def test_load_prices_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prices("/nonexistent/prices.json")
