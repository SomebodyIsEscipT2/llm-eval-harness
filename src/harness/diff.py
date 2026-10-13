"""Compare two run artifacts and report regressions.

This is the feature that turns the harness from "a test runner" into
"the tool that tells you when a model or prompt regressed." Given a
baseline run and a candidate run, it matches cases by index (and shows
the input text) and classifies each as:

  fixed       failed before, passes now
  regressed   passed before, fails now
  still_pass  passed before, passes now
  still_fail  failed before, fails now
  added       only in the candidate
  removed     only in the baseline

Matching by index is deliberate: both runs come from the same eval file
in the order cases were declared, so index is stable across runs. We also
compare the input text and warn if the case at that index changed, so a
mis-aligned diff is visible rather than silently wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.runner import RunResult


@dataclass
class CaseDiff:
    index: int
    status: str  # fixed | regressed | still_pass | still_fail | added | removed
    input: str
    baseline_passed: Optional[bool]
    candidate_passed: Optional[bool]
    baseline_detail: Optional[str] = None
    candidate_detail: Optional[str] = None
    input_changed: bool = False


@dataclass
class DiffResult:
    baseline_model: str
    candidate_model: str
    baseline_suite: str
    candidate_suite: str
    cases: List[CaseDiff] = field(default_factory=list)
    baseline_summary: Dict[str, Any] = field(default_factory=dict)
    candidate_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def regressed(self) -> List[CaseDiff]:
        return [c for c in self.cases if c.status == "regressed"]

    @property
    def fixed(self) -> List[CaseDiff]:
        return [c for c in self.cases if c.status == "fixed"]

    @property
    def still_pass(self) -> List[CaseDiff]:
        return [c for c in self.cases if c.status == "still_pass"]

    @property
    def still_fail(self) -> List[CaseDiff]:
        return [c for c in self.cases if c.status == "still_fail"]

    @property
    def verdict(self) -> str:
        """One word summarising the comparison."""
        if self.regressed:
            return "REGRESSION"
        if self.fixed:
            return "IMPROVED"
        return "UNCHANGED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "baseline": {
                "model": self.baseline_model,
                "suite": self.baseline_suite,
                "summary": self.baseline_summary,
            },
            "candidate": {
                "model": self.candidate_model,
                "suite": self.candidate_suite,
                "summary": self.candidate_summary,
            },
            "counts": {
                "regressed": len(self.regressed),
                "fixed": len(self.fixed),
                "still_pass": len(self.still_pass),
                "still_fail": len(self.still_fail),
            },
            "cases": [c.__dict__ for c in self.cases],
        }


def load_artifact(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Run artifact not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "cases" not in data or "summary" not in data:
        raise ValueError(f"{p} is not a harness run artifact (missing keys)")
    return data


def compare(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> DiffResult:
    """Diff two run-artifact dicts."""
    b_cases = {c["index"]: c for c in baseline["cases"]}
    c_cases = {c["index"]: c for c in candidate["cases"]}
    all_idx = sorted(set(b_cases) | set(c_cases))

    diffs: List[CaseDiff] = []
    for i in all_idx:
        b, c = b_cases.get(i), c_cases.get(i)
        if b is None:
            diffs.append(CaseDiff(i, "added", c["input"], None, c["passed"]))
            continue
        if c is None:
            diffs.append(CaseDiff(i, "removed", b["input"], b["passed"], None))
            continue
        changed = b.get("input") != c.get("input")
        if b["passed"] and not c["passed"]:
            status = "regressed"
        elif not b["passed"] and c["passed"]:
            status = "fixed"
        elif b["passed"] and c["passed"]:
            status = "still_pass"
        else:
            status = "still_fail"
        diffs.append(
            CaseDiff(
                i,
                status,
                c["input"],
                b["passed"],
                c["passed"],
                b.get("detail"),
                c.get("detail"),
                changed,
            )
        )

    return DiffResult(
        baseline_model=baseline.get("model", "?"),
        candidate_model=candidate.get("model", "?"),
        baseline_suite=baseline.get("suite", "?"),
        candidate_suite=candidate.get("suite", "?"),
        cases=diffs,
        baseline_summary=baseline.get("summary", {}),
        candidate_summary=candidate.get("summary", {}),
    )


def _pct(summary: Dict[str, Any]) -> str:
    rate = summary.get("pass_rate")
    if rate is None:
        return "?"
    return f"{rate * 100:.1f}%"


def _truncate(text: str, width: int = 46) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def format_diff(result: DiffResult, *, show_all: bool = False) -> str:
    lines: List[str] = []
    lines.append(f"verdict    {result.verdict}")
    lines.append(
        f"baseline   {result.baseline_model}  "
        f"({_pct(result.baseline_summary)})"
    )
    lines.append(
        f"candidate  {result.candidate_model}  "
        f"({_pct(result.candidate_summary)})"
    )
    counts = (
        f"regressed  {len(result.regressed)}   "
        f"fixed  {len(result.fixed)}   "
        f"still pass  {len(result.still_pass)}   "
        f"still fail  {len(result.still_fail)}"
    )
    lines.append(counts)

    def block(title: str, cases: List[CaseDiff]) -> None:
        if not cases:
            return
        lines.append("")
        lines.append(title)
        for c in cases:
            lines.append(f"  #{c.index:<3} {_truncate(c.input)}")
            if c.status in ("regressed", "still_fail") and c.candidate_detail:
                lines.append(f"        ↳ {c.candidate_detail}")
            if c.input_changed:
                lines.append("        ⚠ input changed between runs")

    block("REGRESSED (passed before, fails now)", result.regressed)
    block("FIXED (failed before, passes now)", result.fixed)
    if show_all:
        block("STILL FAILING", result.still_fail)
        block("STILL PASSING", result.still_pass)

    misaligned = [c for c in result.cases if c.input_changed]
    if misaligned:
        lines.append("")
        lines.append(
            f"⚠ {len(misaligned)} case(s) had different input text between "
            "runs; the comparison may be mis-aligned."
        )
    return "\n".join(lines)
