"""Scorer registry."""

from __future__ import annotations

from typing import Any, Dict

from harness.scorers.base import Contains, ExactMatch, Regex, Scorer

_BUILTINS = {
    "exact": ExactMatch,
    "contains": Contains,
    "regex": Regex,
}


def available() -> Dict[str, type]:
    return dict(_BUILTINS)


def build(spec: Any) -> Scorer:
    """Turn a compact spec into a Scorer.

    Spec forms accepted:
      "exact"                       -> ExactMatch(case_sensitive=True)
      "exact:ci"                    -> ExactMatch(case_sensitive=False)
      {"type": "contains", ...}     -> keyword args passed to the class
    """
    if isinstance(spec, str):
        name, _, mod = spec.partition(":")
        cls = _BUILTINS.get(name)
        if cls is None:
            raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(_BUILTINS)}")
        kwargs: Dict[str, Any] = {}
        if mod == "ci":
            kwargs["case_sensitive"] = False
        return cls(**kwargs)
    if isinstance(spec, dict):
        data = dict(spec)
        name = data.pop("type")
        if name == "regex":
            return Regex(**data)
        cls = _BUILTINS.get(name)
        if cls is None:
            raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(_BUILTINS)}")
        return cls(**data)
    raise TypeError(f"Unsupported scorer spec: {spec!r}")
