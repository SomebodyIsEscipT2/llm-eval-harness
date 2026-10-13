"""LLM-as-judge scoring.

Some outputs can't be checked with a string comparison — "is this summary
accurate?", "is this answer helpful?". For those we ask a model to grade,
with a fixed rubric and a **fixed judge model** so runs stay comparable.

Two rules keep this honest:

1. **The judge model is pinned.** Comparing runs graded by different judges
   is meaningless, so the judge is configured explicitly and recorded in the
   run artifact.
2. **The judge returns structured output.** We ask for JSON and fail the run
   loudly if we can't parse a verdict — never silently pass.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

from harness.providers.base import Provider, ProviderError
from harness.scorers.base import Score, Scorer
from harness.scorers.json_schema import extract_json

JUDGE_TEMPLATE = """You are grading one answer for an evaluation.

Task given to the model:
{prompt}

{expected_block}
Model's answer:
{output}

Grade the answer on a 0.0-1.0 scale, where 1.0 is fully correct and 0.0 is
wrong or unhelpful. Then give a one-sentence reason.

Respond with ONLY this JSON and nothing else:
{{"score": <number 0.0-1.0>, "reason": "<one sentence>"}}"""


class LlmJudge(Scorer):
    """Grade an output with a model, using a fixed rubric.

    The scorer is constructed with the provider and judge model to use; the
    case's own prompt is passed through at score time via ``kwargs['prompt']``.
    """

    name = "llm_judge"

    def __init__(
        self,
        provider: Optional[Provider] = None,
        *,
        model: str = "openai/gpt-4o-mini",
        threshold: float = 0.7,
    ) -> None:
        self.provider = provider
        self.model = model
        self.threshold = threshold

    def _build_prompt(self, prompt: str, output: str, expected: Optional[str]) -> str:
        if expected:
            expected_block = f"Reference answer (may be one valid answer among several):\n{expected}\n"
        else:
            expected_block = "(No reference answer; judge on correctness and helpfulness.)\n"
        return JUDGE_TEMPLATE.format(
            prompt=prompt, expected_block=expected_block, output=output
        )

    def _parse_verdict(self, text: str) -> Score:
        candidate = extract_json(text)
        if candidate is None:
            return Score(False, 0.0, "judge returned no JSON verdict")
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return Score(False, 0.0, "judge returned malformed JSON")
        try:
            value = float(data["score"])
        except (KeyError, TypeError, ValueError):
            return Score(False, 0.0, "judge verdict missing a numeric 'score'")
        value = max(0.0, min(1.0, value))
        reason = str(data.get("reason", "")).strip()
        return Score(value >= self.threshold, value, reason or None)

    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        raise NotImplementedError(
            "LlmJudge is asynchronous and network-bound; use ascore()."
        )

    async def ascore(
        self,
        output: str,
        expected: Optional[str] = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> Score:
        if self.provider is None:
            return Score(False, 0.0, "llm_judge requires a provider")
        judge_prompt = self._build_prompt(prompt, output, expected)
        try:
            completion = await self.provider.complete(
                judge_prompt, model=self.model, temperature=0.0
            )
        except ProviderError as exc:
            return Score(False, 0.0, f"judge call failed: {exc}")
        return self._parse_verdict(completion.text)
