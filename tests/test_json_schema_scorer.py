"""Tests for the JSON-schema scorer and its extraction helpers."""

import pytest

from harness.scorers.json_schema import JsonSchema, extract_json, validate


def test_extract_plain_object():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_from_fence():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
    assert extract_json(text) == '{"a": 1}'


def test_extract_from_prose():
    text = 'The result is {"a": 1} as requested.'
    assert extract_json(text) == '{"a": 1}'


def test_extract_array():
    assert extract_json("[1, 2, 3]") == "[1, 2, 3]"


def test_extract_none_when_no_json():
    assert extract_json("just words") is None


def test_validate_required_missing():
    errors = validate({}, {"type": "object", "required": ["name"]})
    assert any("name" in e for e in errors)


def test_validate_type_mismatch():
    errors = validate("nope", {"type": "number"})
    assert errors


def test_validate_bool_is_not_number():
    errors = validate(True, {"type": "number"})
    assert errors


def test_validate_nested_properties():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    assert validate({"items": ["a", "b"]}, schema) == []
    assert validate({"items": ["a", 2]}, schema)


def test_validate_enum():
    schema = {"type": "string", "enum": ["low", "high"]}
    assert validate("low", schema) == []
    assert validate("medium", schema)


def test_scorer_pass_with_schema():
    s = JsonSchema({"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}})
    result = s.score('{"name": "Escipion"}')
    assert result.passed
    assert result.score == 1.0


def test_scorer_fail_on_missing_field():
    s = JsonSchema({"type": "object", "required": ["name"]})
    result = s.score('{"other": 1}')
    assert not result.passed
    assert "name" in (result.detail or "")


def test_scorer_fail_on_invalid_json():
    s = JsonSchema({})
    result = s.score("{not json}")
    assert not result.passed
    assert "invalid JSON" in (result.detail or "")


def test_scorer_no_json_at_all():
    s = JsonSchema({})
    result = s.score("plain prose with no structure")
    assert not result.passed
    assert "no JSON" in (result.detail or "")


def test_scorer_parses_fenced_output():
    s = JsonSchema({"type": "object", "required": ["ok"]})
    result = s.score('Sure!\n```json\n{"ok": true}\n```')
    assert result.passed


def test_scorer_from_registry():
    from harness.scorers import build

    s = build({"type": "json_schema", "schema": {"type": "object"}})
    assert isinstance(s, JsonSchema)
    assert s.score('{"anything": 1}').passed
