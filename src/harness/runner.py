"""Async runner: execute cases against a provider, record structured results."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from harness.eval_file import Case, EvalSuite
from harness.pricing import estimate
from harness.providers.base import Provider, ProviderError, Usage
from harness.scorers import build as build_scorer


@dataclass
class CaseResult:
    index: int
    input: str
    expected: Optional[str]
    output: Optional[str]
    passed: bool
    score: float
    detail: Optional[str]
    latency_ms: float
    usage: Usage
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d


@dataclass
class RunResult:
    suite: str
    model: str
    provider: str
    started_at: float
    cases: List[CaseResult]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for c in self.cases:
            total = total + c.usage
        return total

    @property
    def avg_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.latency_ms for c in self.cases) / len(self.cases)

    @property
    def total_cost_usd(self) -> Optional[float]:
        """Sum of per-case costs, or None if the model isn't priced."""
        costs = [c.cost_usd for c in self.cases if c.cost_usd is not None]
        if not costs:
            return None
        return sum(costs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "model": self.model,
            "provider": self.provider,
            "started_at": self.started_at,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.total - self.passed,
                "pass_rate": self.pass_rate,
                "avg_latency_ms": self.avg_latency_ms,
                "usage": asdict(self.total_usage),
                "total_cost_usd": self.total_cost_usd,
            },
            "cases": [c.to_dict() for c in self.cases],
        }


async def _run_case(
    index: int,
    case: Case,
    provider: Provider,
    model: Optional[str],
    sem: asyncio.Semaphore,
    retries: int,
    prices: Optional[Dict[str, Any]] = None,
) -> CaseResult:
    scorer = build_scorer(case.scorer)
    async with sem:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                completion = await provider.complete(
                    case.input, system=case.system, model=model
                )
                verdict = scorer.score(completion.text, case.expected, **case.kwargs)
                cost = estimate(completion.model, completion.usage, prices)
                return CaseResult(
                    index=index,
                    input=case.input,
                    expected=case.expected,
                    output=completion.text,
                    passed=verdict.passed,
                    score=verdict.score,
                    detail=verdict.detail,
                    latency_ms=completion.latency_ms,
                    usage=completion.usage,
                    tags=case.tags,
                    cost_usd=cost,
                )
            except ProviderError as exc:
                last_err = exc
                if attempt < retries:
                    await asyncio.sleep(0.4 * (2**attempt))
                continue
        return CaseResult(
            index=index,
            input=case.input,
            expected=case.expected,
            output=None,
            passed=False,
            score=0.0,
            detail="provider error",
            latency_ms=0.0,
            usage=Usage(),
            tags=case.tags,
            error=str(last_err),
        )


async def run_suite(
    suite: EvalSuite,
    provider: Provider,
    *,
    model: Optional[str] = None,
    concurrency: int = 4,
    retries: int = 1,
    prices: Optional[Dict[str, Any]] = None,
) -> RunResult:
    """Run every case in ``suite`` against ``provider``."""
    started = time.time()
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        _run_case(i, c, provider, model, sem, retries, prices)
        for i, c in enumerate(suite.cases)
    ]
    results = await asyncio.gather(*tasks)
    return RunResult(
        suite=suite.name,
        model=model or getattr(provider, "model", "unknown"),
        provider=provider.name,
        started_at=started,
        cases=list(results),
    )
