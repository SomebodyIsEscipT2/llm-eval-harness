"""Tests for the LLM-as-judge scorer, using a scripted fake provider."""

import pytest

from harness.providers.base import Completion, Provider, ProviderError, Usage
from harness.scorers.llm_judge import LlmJudge


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None
        self.last_model = None

    async def complete(self, prompt, *, system=None, model=None, **kwargs):
        self.last_prompt = prompt
        self.last_model = model
        if isinstance(self.reply, Exception):
            raise self.reply
        return Completion(self.reply, model or "judge-1", Usage(10, 5, 15), 88.0)


async def test_judge_passes_high_score():
    provider = ScriptedProvider('{"score": 0.9, "reason": "correct and clear"}')
    judge = LlmJudge(provider, model="judge-model", threshold=0.7)
    result = await judge.ascore("4", "4", prompt="What is 2+2?")
    assert result.passed
    assert result.score == 0.9
    assert result.detail == "correct and clear"


async def test_judge_fails_low_score():
    provider = ScriptedProvider('{"score": 0.2, "reason": "wrong"}')
    judge = LlmJudge(provider, threshold=0.7)
    result = await judge.ascore("5", "4", prompt="What is 2+2?")
    assert not result.passed
    assert result.score == 0.2


async def test_judge_clamps_out_of_range():
    provider = ScriptedProvider('{"score": 1.8, "reason": "over"}')
    judge = LlmJudge(provider)
    result = await judge.ascore("x", None, prompt="q")
    assert result.score == 1.0


async def test_judge_handles_fenced_json():
    provider = ScriptedProvider('```json\n{"score": 0.8, "reason": "good"}\n```')
    judge = LlmJudge(provider)
    result = await judge.ascore("x", None, prompt="q")
    assert result.passed


async def test_judge_malformed_verdict_fails_loudly():
    provider = ScriptedProvider("I think it is good, maybe 9/10")
    judge = LlmJudge(provider)
    result = await judge.ascore("x", None, prompt="q")
    assert not result.passed
    assert "JSON" in (result.detail or "")


async def test_judge_missing_score_field():
    provider = ScriptedProvider('{"reason": "no score here"}')
    judge = LlmJudge(provider)
    result = await judge.ascore("x", None, prompt="q")
    assert not result.passed
    assert "score" in (result.detail or "")


async def test_judge_provider_error_is_caught():
    provider = ScriptedProvider(ProviderError("upstream down"))
    judge = LlmJudge(provider)
    result = await judge.ascore("x", None, prompt="q")
    assert not result.passed
    assert "judge call failed" in (result.detail or "")


async def test_judge_uses_pinned_model():
    provider = ScriptedProvider('{"score": 1.0, "reason": "ok"}')
    judge = LlmJudge(provider, model="pinned-judge")
    await judge.ascore("x", None, prompt="q")
    assert provider.last_model == "pinned-judge"


async def test_judge_prompt_includes_expected_when_present():
    provider = ScriptedProvider('{"score": 1.0, "reason": "ok"}')
    judge = LlmJudge(provider)
    await judge.ascore("out", "ref", prompt="question")
    assert "Reference answer" in provider.last_prompt
    assert "ref" in provider.last_prompt


async def test_judge_prompt_notes_absence_of_reference():
    provider = ScriptedProvider('{"score": 1.0, "reason": "ok"}')
    judge = LlmJudge(provider)
    await judge.ascore("out", None, prompt="question")
    assert "No reference answer" in provider.last_prompt


def test_sync_score_raises():
    judge = LlmJudge(None)
    with pytest.raises(NotImplementedError):
        judge.score("x")


def test_registry_build_injects_provider():
    from harness.scorers import build

    provider = ScriptedProvider('{"score": 1.0, "reason": "ok"}')
    judge = build({"type": "llm_judge", "judge_model": "j"}, provider=provider)
    assert isinstance(judge, LlmJudge)
    assert judge.model == "j"
