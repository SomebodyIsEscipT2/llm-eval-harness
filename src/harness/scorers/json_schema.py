"""JSON-schema scoring: does the model's output parse as the shape we asked for?

This is the workhorse scorer for structured-output evaluations. It checks
that the output is valid JSON and optionally that it satisfies a schema.

The schema check is intentionally dependency-free: we implement a focused
subset of JSON Schema that covers what real evals actually assert —
type, required properties, property types, enum, and array item types.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from harness.scorers.base import Score, Scorer

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[str]:
    """Pull a JSON value out of model output.

    Models often wrap JSON in prose or markdown fences. We try, in order:
    the whole string, a fenced block, and a brace/bracket-spanning slice.
    Returns the candidate string or None if nothing looks like JSON.
    """
    text = text.strip()
    if not text:
        return None

    # 1. The whole thing.
    if text[0] in "{[":
        return text

    # 2. A fenced code block.
    m = _FENCE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner[:1] in "{[":
            return inner

    # 3. First {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _check_type(value: Any, want: str) -> bool:
    if want == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if want == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if want == "boolean":
        return isinstance(value, bool)
    expected = _TYPE_MAP.get(want)
    if expected is None:
        return True  # unknown type keyword: don't fail the run
    return isinstance(value, expected)


def validate(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: List[str] = []
    if not isinstance(schema, dict):
        return errors

    want = schema.get("type")
    if want and not _check_type(value, want):
        errors.append(f"{path}: expected {want}, got {type(value).__name__}")
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property missing")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

    return errors


class JsonSchema(Scorer):
    """Pass when the output parses as JSON and satisfies the schema."""

    name = "json_schema"

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        self.schema = schema or {}

    def score(self, output: str, expected: Optional[str] = None, **kwargs: Any) -> Score:
        schema = kwargs.get("schema", self.schema)
        candidate = extract_json(output)
        if candidate is None:
            return Score(False, 0.0, "no JSON value found in output")
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return Score(False, 0.0, f"invalid JSON: {exc.msg} at pos {exc.pos}")
        errors = validate(value, schema) if schema else []
        if errors:
            return Score(False, 0.0, "schema: " + "; ".join(errors[:3]))
        return Score(True, 1.0)
