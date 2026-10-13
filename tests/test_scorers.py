"""Tests for the scorer implementations."""

from harness.scorers import build
from harness.scorers.base import Contains, ExactMatch, Regex


def test_exact_match_pass():
    s = ExactMatch()
    assert s.score("4", "4").passed


def test_exact_match_strips_whitespace():
    s = ExactMatch()
    assert s.score("  4\n", "4").passed


def test_exact_match_case_sensitive_fails():
    s = ExactMatch(case_sensitive=True)
    assert not s.score("Yes", "yes").passed


def test_exact_match_case_insensitive_passes():
    s = ExactMatch(case_sensitive=False)
    assert s.score("Yes", "yes").passed


def test_exact_match_requires_expected():
    s = ExactMatch()
    result = s.score("anything")
    assert not result.passed
    assert "expected" in (result.detail or "")


def test_contains_single():
    s = Contains()
    assert s.score("The answer is 42.", "42").passed


def test_contains_missing_reports_detail():
    s = Contains()
    result = s.score("nothing here", "42")
    assert not result.passed
    assert "42" in (result.detail or "")


def test_contains_multiple_needles():
    s = Contains()
    assert s.score("alpha and beta", contains=["alpha", "beta"]).passed
    assert not s.score("alpha only", contains=["alpha", "beta"]).passed


def test_regex_match():
    s = Regex(r"\d{4}-\d{2}-\d{2}")
    assert s.score("date: 2026-10-12").passed


def test_regex_no_match():
    s = Regex(r"^yes$")
    result = s.score("no")
    assert not result.passed


def test_build_from_string():
    assert isinstance(build("exact"), ExactMatch)
    assert isinstance(build("exact:ci"), ExactMatch)


def test_build_from_dict():
    s = build({"type": "contains", "case_sensitive": False})
    assert isinstance(s, Contains)


def test_build_regex_from_dict():
    s = build({"type": "regex", "pattern": "abc"})
    assert isinstance(s, Regex)


def test_build_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        build("does-not-exist")
