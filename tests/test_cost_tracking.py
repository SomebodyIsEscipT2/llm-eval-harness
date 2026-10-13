"""Tests for cost tracking through the run pipeline."""

from harness.eval_file import Case, EvalSuite
from harness.pricing import DEFAULT_PRICES
from harness.providers.base import Completion, Provider, Usage
from harness.runner import run_suite


class CostedProvider(Provider):
    name = "costed"

    async def complete(self, prompt, *, system=None, model=None, **kwargs):
        return Completion(
            text="4",
            model=model or "openai/gpt-4o-mini",
            usage=Usage(1_000_000, 1_000_000, 2_000_000),
            latency_ms=12.0,
        )


async def test_cost_recorded_per_case():
    suite = EvalSuite(name="t", cases=[Case(input="a", expected="4")])
    result = await run_suite(suite, CostedProvider(), model="openai/gpt-4o-mini")
    # 1M prompt @ 0.15 + 1M completion @ 0.60 = 0.75
    assert result.cases[0].cost_usd == 0.75
    assert result.total_cost_usd == 0.75


async def test_unpriced_model_yields_no_cost():
    suite = EvalSuite(name="t", cases=[Case(input="a", expected="4")])
    result = await run_suite(suite, CostedProvider(), model="unknown/model-xyz")
    assert result.cases[0].cost_usd is None
    assert result.total_cost_usd is None


async def test_cost_appears_in_artifact():
    suite = EvalSuite(name="t", cases=[Case(input="a", expected="4")])
    result = await run_suite(suite, CostedProvider(), model="openai/gpt-4o-mini")
    summary = result.to_dict()["summary"]
    assert summary["total_cost_usd"] == 0.75
