"""Scorers: turn a completion into a pass/fail verdict plus a score."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Score:
    """Result of judging one completion.

    ``passed`` is the headline verdict; ``score`` is a 0..1 value for
    partial credit; ``detail`` explains a failure for the report.
    """

    passed: bool
    score: float = 1.0
    detail: Optional[str] = None


class Scorer(abc.ABC):
    """Base class for every scoring strategy.

    ``expected`` is optional because some scorers (e.g. JSON validity)
    judge the output on its own terms rather than against a reference.
    """

    name: str = "scorer"

    @abc.abstractmethod
    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        raise NotImplementedError


class ExactMatch(Scorer):
    """Strip and compare, optionally case-insensitively."""

    name = "exact"

    def __init__(self, case_sensitive: bool = True) -> None:
        self.case_sensitive = case_sensitive

    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        if expected is None:
            return Score(False, 0.0, "exact match requires an expected value")
        a, b = output.strip(), expected.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        ok = a == b
        return Score(ok, 1.0 if ok else 0.0, None if ok else "output differs from expected")


class Contains(Scorer):
    """Pass if a required substring (or all of several) is present."""

    name = "contains"

    def __init__(self, *, case_sensitive: bool = True) -> None:
        self.case_sensitive = case_sensitive

    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        needles = kwargs.get("contains", expected)
        if needles is None:
            return Score(False, 0.0, "contains requires a needle")
        if isinstance(needles, str):
            needles = [needles]
        hay = output if self.case_sensitive else output.lower()
        missing = []
        for n in needles:
            target = n if self.case_sensitive else n.lower()
            if target not in hay:
                missing.append(n)
        ok = not missing
        if ok:
            return Score(True, 1.0)
        return Score(False, 0.0, f"missing: {', '.join(map(repr, missing))}")


class Regex(Scorer):
    """Pass if the output matches a regular expression."""

    name = "regex"

    def __init__(self, pattern: Optional[str] = None) -> None:
        if pattern is None:
            raise ValueError("Regex scorer requires a pattern")
        import re

        self._re = re.compile(pattern)

    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        m = self._re.search(output)
        if m:
            return Score(True, 1.0)
        return Score(False, 0.0, f"no match for /{self._re.pattern}/")
