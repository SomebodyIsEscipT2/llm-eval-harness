# llm-eval-harness

[![CI](https://github.com/SomebodyIsEscipT2/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/SomebodyIsEscipT2/llm-eval-harness/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Run structured evaluations against any LLM provider — and see what changed.**

`llm-eval-harness` lets you define test cases for an LLM, run them against a
provider, and get a clear report: pass rate, cost, latency, and (soon)
regression diffs between model versions. Static evals you can put in CI.

## Why

Model quality drifts. A prompt that worked yesterday can regress when a
provider updates a model. Swapping models is a decision that needs evidence.
This harness makes "is model B better than model A on *my* task?" a
one-command answer instead of a vibe.

## Quick start

```bash
pip install -e ".[dev]"

export OPENROUTER_API_KEY=sk-or-...

harness evals/arithmetic.yaml --model openrouter/openai/gpt-4o-mini
```

Example output:

```
suite      arithmetic-basics
provider   openrouter
model      openrouter/openai/gpt-4o-mini
pass rate  5/5 (100.0%)
latency    412 ms avg
tokens     186 total (95 prompt / 91 completion)

  #  STATUS   SCORE       MS  INPUT
-----------------------------------------------
  0  PASS      1.00      380  What is 2 + 2? Reply with just the number.
  1  PASS      1.00      401  What is 7 * 6? Reply with just the number.
  ...
```

Write a JSON artifact for later comparison:

```bash
harness evals/arithmetic.yaml --json runs/baseline.json
```

## Defining an eval

YAML or JSON. Each case has an input, an optional expected value, and a scorer.

```yaml
name: arithmetic-basics
defaults:
  scorer: exact:ci          # case-insensitive exact match
cases:
  - input: "What is 2 + 2? Reply with just the number."
    expected: "4"
    tags: [math]
  - input: "Is 17 prime? Reply yes or no."
    expected: "yes"
    tags: [math, reasoning]
```

### Scorers (v0.1)

| Scorer | Spec | Passes when |
| --- | --- | --- |
| Exact match | `exact` / `exact:ci` | Output equals expected (optionally case-insensitive) |
| Contains | `{type: contains}` | Output contains the needle(s) |
| Regex | `{type: regex, pattern: "..."}` | Output matches the pattern |

## Architecture

```
src/harness/
  providers/     one interface, many backends (openrouter built in)
  scorers/       exact · contains · regex   (json-schema · llm-judge planned)
  eval_file.py   YAML/JSON suite loading
  runner.py      async execution, retries, token/latency capture
  report.py      summary + per-case tables
  cli.py         the `harness` command
```

Design notes:

- **Providers stay dumb.** A provider returns a completion; it never scores,
  retries, or persists. That keeps adding a backend to ~50 lines.
- **Scorers stay pure.** Given output and expected, return pass/fail + score.
  No network. Trivially testable.
- **The runner owns the hard parts.** Concurrency, retries with backoff, and
  recording usage so a run is reproducible.

## Roadmap

- [x] Provider abstraction + OpenRouter
- [x] Scorers: exact, contains, regex
- [x] Async runner with token/latency capture
- [x] CLI report
- [ ] JSON-schema scorer
- [ ] LLM-as-judge scorer (fixed judge model for comparability)
- [ ] `harness diff runA.json runB.json` — regression detection
- [ ] OpenAI-compatible + local (Ollama) providers
- [ ] Cost estimation from model price tables

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests mock all network calls, so the suite runs offline and in CI.

## License

MIT — see [LICENSE](LICENSE).

## Scorers

| Scorer | Spec | Passes when |
| --- | --- | --- |
| Exact match | `exact` / `exact:ci` | Output equals expected (optionally case-insensitive) |
| Contains | `{type: contains, case_sensitive: false}` | Output contains the needle(s) |
| Regex | `{type: regex, pattern: "..."}` | Output matches the pattern |
| JSON schema | `{type: json_schema, schema: {...}}` | Output parses as JSON and satisfies the schema |
| LLM judge | `{type: llm_judge, judge_model: "...", threshold: 0.7}` | A pinned judge model grades the answer at or above the threshold |

### Structured output

Ask for JSON and assert its shape, without an extra dependency:

```yaml
defaults:
  scorer:
    type: json_schema
    schema:
      type: object
      required: [title, priority]
      properties:
        title: {type: string}
        priority: {type: string, enum: [low, medium, high]}
```

The scorer pulls JSON out of fences or prose, then checks `type`,
`required`, `properties`, `items`, and `enum`. No network, no dependency.

### Graded (open-ended) answers

For summaries and other answers with no single correct string, use a judge —
with a **pinned model** so runs stay comparable, and structured output so a
missing verdict fails loudly instead of silently passing:

```yaml
defaults:
  scorer:
    type: llm_judge
    judge_model: openai/gpt-4o-mini
    threshold: 0.7
```

## TLS-intercepting environments

If your network intercepts TLS (corporate proxy, egress gateway), Python's
SSL stack may reject the chain even where `curl` works. Point the harness at
a CA bundle, or disable verification for a trusted local proxy:

```bash
HARNESS_TLS_VERIFY=/path/to/ca-bundle.pem harness evals/arithmetic.yaml
HARNESS_TLS_VERIFY=0 harness evals/arithmetic.yaml   # trusted local proxy only
```
