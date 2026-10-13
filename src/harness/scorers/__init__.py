"""Scorer registry."""

from __future__ import annotations

from typing import Any, Dict

from harness.scorers.base import Contains, ExactMatch, Regex, Scorer
from harness.scorers.json_schema import JsonSchema
from harness.scorers.llm_judge import LlmJudge

_BUILTINS = {
    "exact": ExactMatch,
    "contains": Contains,
    "regex": Regex,
    "json_schema": JsonSchema,
    "llm_judge": LlmJudge,
}

# Scorers that must call out to a provider and therefore only work through
# the async path.
ASYNC_SCORERS = {"llm_judge"}


def available() -> Dict[str, type]:
    return dict(_BUILTINS)


def is_async(spec: Any) -> bool:
    if isinstance(spec, str):
        return spec.partition(":")[0] in ASYNC_SCORERS
    if isinstance(spec, dict):
        return spec.get("type") in ASYNC_SCORERS
    return False


def build(spec: Any, **extra: Any) -> Scorer:
    """Turn a compact spec into a Scorer.

    Spec forms accepted:
      "exact"                       -> ExactMatch(case_sensitive=True)
      "exact:ci"                    -> ExactMatch(case_sensitive=False)
      "json_schema"                 -> JsonSchema({})
      {"type": "contains", ...}     -> keyword args passed to the class
      {"type": "llm_judge", ...}    -> LlmJudge(provider=..., **rest)

    ``extra`` (e.g. ``provider=``, ``judge_model=``) is merged for scorers
    that need runtime objects the eval file cannot express.
    """
    if isinstance(spec, str):
        name, _, mod = spec.partition(":")
        cls = _BUILTINS.get(name)
        if cls is None:
            raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(_BUILTINS)}")
        kwargs: Dict[str, Any] = {}
        if mod == "ci":
            kwargs["case_sensitive"] = False
        if name == "llm_judge":
            kwargs.update(extra)
        if name == "json_schema" and "schema" in extra:
            kwargs["schema"] = extra["schema"]
        return cls(**kwargs)

    if isinstance(spec, dict):
        data = dict(spec)
        name = data.pop("type")
        cls = _BUILTINS.get(name)
        if cls is None:
            raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(_BUILTINS)}")
        if name == "llm_judge":
            data.update({k: v for k, v in extra.items() if k == "provider" or k == "model"})
            data.setdefault("model", data.pop("judge_model", "openai/gpt-4o-mini"))
        return cls(**data)

    raise TypeError(f"Unsupported scorer spec: {spec!r}")
