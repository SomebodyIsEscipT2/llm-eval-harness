"""Tests for the regression diff."""

import json

import pytest

from harness.diff import compare, format_diff, load_artifact


def _artifact(model, suite, cases, pass_rate):
    passed = sum(1 for c in cases if c["passed"])
    return {
        "suite": suite,
        "model": model,
        "provider": "fake",
        "started_at": 0.0,
        "summary": {
            "total": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
            "pass_rate": pass_rate,
            "avg_latency_ms": 10.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        "cases": cases,
    }


def _case(i, inp, passed, detail=None):
    return {
        "index": i,
        "input": inp,
        "expected": None,
        "output": "x",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": detail,
        "latency_ms": 10.0,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "tags": [],
        "error": None,
    }


def test_clean_regression_detected():
    base = _artifact("m-a", "s", [_case(0, "a", True), _case(1, "b", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "a", True), _case(1, "b", False, "wrong")], 0.5)
    result = compare(base, cand)
    assert result.verdict == "REGRESSION"
    assert len(result.regressed) == 1
    assert result.regressed[0].index == 1
    assert len(result.still_pass) == 1


def test_improvement_detected():
    base = _artifact("m-a", "s", [_case(0, "a", False), _case(1, "b", True)], 0.5)
    cand = _artifact("m-b", "s", [_case(0, "a", True), _case(1, "b", True)], 1.0)
    result = compare(base, cand)
    assert result.verdict == "IMPROVED"
    assert len(result.fixed) == 1


def test_unchanged_when_identical():
    base = _artifact("m-a", "s", [_case(0, "a", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "a", True)], 1.0)
    result = compare(base, cand)
    assert result.verdict == "UNCHANGED"


def test_regression_trumps_improvement_in_verdict():
    base = _artifact("m-a", "s", [_case(0, "a", True), _case(1, "b", False)], 0.5)
    cand = _artifact("m-b", "s", [_case(0, "a", False), _case(1, "b", True)], 0.5)
    result = compare(base, cand)
    assert result.verdict == "REGRESSION"
    assert len(result.regressed) == 1
    assert len(result.fixed) == 1


def test_still_failing_classified():
    base = _artifact("m-a", "s", [_case(0, "a", False)], 0.0)
    cand = _artifact("m-b", "s", [_case(0, "a", False)], 0.0)
    result = compare(base, cand)
    assert len(result.still_fail) == 1


def test_added_and_removed_cases():
    base = _artifact("m-a", "s", [_case(0, "a", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "a", True), _case(1, "new", True)], 1.0)
    result = compare(base, cand)
    statuses = {c.index: c.status for c in result.cases}
    assert statuses[1] == "added"

    base2 = _artifact("m-a", "s", [_case(0, "a", True), _case(1, "b", True)], 1.0)
    cand2 = _artifact("m-b", "s", [_case(0, "a", True)], 1.0)
    result2 = compare(base2, cand2)
    statuses2 = {c.index: c.status for c in result2.cases}
    assert statuses2[1] == "removed"


def test_input_change_flagged():
    base = _artifact("m-a", "s", [_case(0, "original", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "DIFFERENT", True)], 1.0)
    result = compare(base, cand)
    assert result.cases[0].input_changed


def test_format_diff_shows_verdict_and_counts():
    base = _artifact("m-a", "s", [_case(0, "a", True), _case(1, "b", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "a", True), _case(1, "b", False, "wrong")], 0.5)
    text = format_diff(compare(base, cand))
    assert "REGRESSION" in text
    assert "regressed  1" in text
    assert "wrong" in text


def test_to_dict_roundtrips():
    base = _artifact("m-a", "s", [_case(0, "a", True)], 1.0)
    cand = _artifact("m-b", "s", [_case(0, "a", False)], 0.0)
    d = compare(base, cand).to_dict()
    assert d["verdict"] == "REGRESSION"
    assert d["counts"]["regressed"] == 1


def test_load_artifact_rejects_non_artifact(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"hello": "world"}')
    with pytest.raises(ValueError):
        load_artifact(p)


def test_load_artifact_missing_file():
    with pytest.raises(FileNotFoundError):
        load_artifact("/nonexistent/run.json")


def test_load_artifact_roundtrip(tmp_path):
    art = _artifact("m", "s", [_case(0, "a", True)], 1.0)
    p = tmp_path / "run.json"
    p.write_text(json.dumps(art))
    loaded = load_artifact(p)
    assert loaded["model"] == "m"
