"""Reporting: turn a RunResult into a CLI table."""

from __future__ import annotations

from typing import List

from harness.runner import RunResult


def _truncate(text: str, width: int = 48) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def summary_table(result: RunResult) -> str:
    u = result.total_usage
    lines: List[str] = []
    lines.append(f"suite      {result.suite}")
    lines.append(f"provider   {result.provider}")
    lines.append(f"model      {result.model}")
    lines.append(
        f"pass rate  {result.passed}/{result.total} "
        f"({result.pass_rate * 100:.1f}%)"
    )
    lines.append(f"latency    {result.avg_latency_ms:.0f} ms avg")
    lines.append(
        f"tokens     {u.total_tokens} total "
        f"({u.prompt_tokens} prompt / {u.completion_tokens} completion)"
    )
    return "\n".join(lines)


def case_table(result: RunResult, *, only_failures: bool = False) -> str:
    rows = [c for c in result.cases if not only_failures or not c.passed]
    if not rows:
        return "(no cases to show)"
    header = f"{'#':>3}  {'STATUS':<6}  {'SCORE':>5}  {'MS':>7}  INPUT"
    sep = "-" * len(header)
    out = [header, sep]
    for c in rows:
        status = "PASS" if c.passed else "FAIL"
        out.append(
            f"{c.index:>3}  {status:<6}  {c.score:>5.2f}  {c.latency_ms:>7.0f}  "
            f"{_truncate(c.input)}"
        )
        if not c.passed and c.detail:
            out.append(f"{'':>3}  {'':<6}  {'':>5}  {'':>7}  \u21b3 {c.detail}")
    return "\n".join(out)


def report(result: RunResult, *, only_failures: bool = False) -> str:
    return summary_table(result) + "\n\n" + case_table(result, only_failures=only_failures)
