"""Eval definitions and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Case:
    """One thing to test a model on."""

    input: str
    expected: Optional[str] = None
    system: Optional[str] = None
    scorer: Any = "exact"
    tags: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuite:
    """A named collection of cases."""

    name: str
    cases: List[Case]
    description: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)


def _load_raw(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_suite(path: str | Path) -> EvalSuite:
    """Load a suite from a JSON or YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")
    data = _load_raw(p)

    defaults = data.get("defaults", {}) or {}
    name = data.get("name") or p.stem
    description = data.get("description", "")

    cases: List[Case] = []
    for i, raw in enumerate(data.get("cases", [])):
        if "input" not in raw:
            raise ValueError(f"Case #{i} in {p} is missing 'input'")
        cases.append(
            Case(
                input=raw["input"],
                expected=raw.get("expected", defaults.get("expected")),
                system=raw.get("system", defaults.get("system")),
                scorer=raw.get("scorer", defaults.get("scorer", "exact")),
                tags=list(raw.get("tags", []) or []),
                kwargs=dict(raw.get("kwargs", {}) or {}),
            )
        )
    if not cases:
        raise ValueError(f"No cases found in {p}")
    return EvalSuite(name=name, cases=cases, description=description, defaults=defaults)
