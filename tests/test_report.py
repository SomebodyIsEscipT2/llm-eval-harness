"""Tests for the reporting layer."""

from harness.providers.base import Usage
from harness.runner import CaseResult, RunResult


def _result():
    return RunResult(
        suite="demo",
        model="fake-1",
        provider="fake",
        started_at=0.0,
        cases=[
            CaseResult(0, "2+2?", "4", "4", True, 1.0, None, 10.0, Usage(1, 1, 2), ["math"]),
            CaseResult(1, "7*6?", "42", "41", False, 0.0, "output differs", 12.0, Usage(1, 1, 2), ["math"]),
        ],
    )


def test_summary_contains_pass_rate():
    from harness.report import summary_table

    text = summary_table(_result())
    assert "1/2" in text
    assert "50.0%" in text


def test_case_table_marks_failures():
    from harness.report import case_table

    text = case_table(_result())
    assert "PASS" in text
    assert "FAIL" in text
    assert "output differs" in text


def test_failures_only_hides_passes():
    from harness.report import case_table

    text = case_table(_result(), only_failures=True)
    assert "PASS" not in text
    assert "FAIL" in text


def test_to_dict_roundtrips():
    d = _result().to_dict()
    assert d["summary"]["total"] == 2
    assert d["summary"]["passed"] == 1
    assert len(d["cases"]) == 2
