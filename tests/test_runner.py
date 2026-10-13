"""Tests for the async runner, using a fake provider."""

from harness.eval_file import Case, EvalSuite
from harness.providers.base import Completion, Provider, Usage
from harness.runner import run_suite


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    async def complete(self, prompt, *, system=None, model=None, **kwargs):
        self.calls += 1
        text = self.answers.get(prompt, "unknown")
        return Completion(
            text=text,
            model=model or "fake-1",
            usage=Usage(1, 1, 2),
            latency_ms=5.0,
        )


async def test_run_suite_counts_passes():
    suite = EvalSuite(
        name="t",
        cases=[
            Case(input="a", expected="1", scorer="exact"),
            Case(input="b", expected="2", scorer="exact"),
        ],
    )
    provider = FakeProvider({"a": "1", "b": "wrong"})
    result = await run_suite(suite, provider)
    assert result.total == 2
    assert result.passed == 1
    assert result.pass_rate == 0.5


async def test_run_suite_all_pass():
    suite = EvalSuite(
        name="t", cases=[Case(input="a", expected="1", scorer="exact")]
    )
    provider = FakeProvider({"a": "1"})
    result = await run_suite(suite, provider)
    assert result.passed == result.total
    assert result.total_usage.total_tokens == 2


class FailingProvider(Provider):
    name = "failing"

    async def complete(self, prompt, *, system=None, model=None, **kwargs):
        from harness.providers.base import ProviderError

        raise ProviderError("always fails")


async def test_provider_errors_become_failed_cases():
    suite = EvalSuite(
        name="t", cases=[Case(input="a", expected="1", scorer="exact")]
    )
    result = await run_suite(suite, FailingProvider(), retries=0)
    assert result.passed == 0
    assert result.cases[0].error
