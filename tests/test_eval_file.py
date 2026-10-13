"""Tests for eval file loading."""

import json

import pytest

from harness.eval_file import load_suite


def test_load_json(tmp_path):
    data = {
        "name": "demo",
        "cases": [{"input": "2+2?", "expected": "4", "tags": ["math"]}],
    }
    p = tmp_path / "demo.json"
    p.write_text(json.dumps(data))
    suite = load_suite(p)
    assert suite.name == "demo"
    assert suite.cases[0].scorer == "exact"
    assert suite.cases[0].tags == ["math"]


def test_load_yaml_with_defaults(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text(
        "name: y\n"
        "defaults:\n"
        "  scorer: exact:ci\n"
        "cases:\n"
        "  - input: hello\n"
        "    expected: hi\n"
    )
    suite = load_suite(p)
    assert suite.cases[0].scorer == "exact:ci"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_suite("/nonexistent/path/eval.yaml")


def test_case_without_input_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: b\ncases:\n  - expected: 4\n")
    with pytest.raises(ValueError):
        load_suite(p)


def test_empty_cases_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("name: e\ncases: []\n")
    with pytest.raises(ValueError):
        load_suite(p)
