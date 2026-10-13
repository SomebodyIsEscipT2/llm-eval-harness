"""CLI entrypoint: run an eval suite, or diff two run artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from harness.diff import compare, format_diff, load_artifact
from harness.eval_file import load_suite
from harness.report import report
from harness.runner import run_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness",
        description="Run structured evaluations against an LLM provider, and diff runs.",
    )
    sub = p.add_subparsers(dest="command")

    # --- run (default) ---
    run = sub.add_parser("run", help="Run an eval suite")
    run.add_argument("eval_file", help="Path to a .json/.yaml eval suite")
    run.add_argument("--provider", default="openrouter")
    run.add_argument("--model", default=None)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--json", dest="json_out", default=None)
    run.add_argument("--failures-only", action="store_true")
    run.add_argument("--costs", default=None, help="JSON price-table overlay")

    # --- diff ---
    diff = sub.add_parser("diff", help="Compare two run artifacts")
    diff.add_argument("baseline")
    diff.add_argument("candidate")
    diff.add_argument("--show-all", action="store_true", help="Include unchanged cases")
    diff.add_argument("--json", dest="json_out", default=None)
    diff.add_argument(
        "--fail-on-regression",
        dest="fail_on_regression",
        action="store_true",
        default=True,
        help="Exit non-zero when a regression is found (default)",
    )
    diff.add_argument(
        "--no-fail-on-regression",
        dest="fail_on_regression",
        action="store_false",
    )

    p.add_argument("legacy", nargs="?", help=argparse.SUPPRESS)
    return p


def _normalize_argv(argv: Optional[list]) -> Optional[list]:
    """Backwards compatibility: `harness eval.yaml` == `harness run eval.yaml`.

    If no known subcommand is given first, and the first argument is a file
    path, inject `run`.
    """
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    known = {"run", "diff"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv = ["run"] + argv
    return argv


async def _arun(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.eval_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from harness.providers import get as get_provider
    from harness.providers.base import ProviderError
    from harness.pricing import load_prices

    try:
        provider_cls = get_provider(args.provider)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        provider = provider_cls()
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        prices = load_prices(getattr(args, "costs", None))
    except FileNotFoundError as exc:
        await provider.aclose()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = await run_suite(
            suite,
            provider,
            model=args.model,
            concurrency=args.concurrency,
            retries=args.retries,
            prices=prices,
        )
    finally:
        await provider.aclose()

    print(report(result, only_failures=args.failures_only))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nartifact written to {args.json_out}")

    return 0 if result.passed == result.total else 1


def _run_diff(args: argparse.Namespace) -> int:
    try:
        baseline = load_artifact(args.baseline)
        candidate = load_artifact(args.candidate)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = compare(baseline, candidate)
    print(format_diff(result, show_all=args.show_all))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\ndiff written to {args.json_out}")

    if result.regressed and args.fail_on_regression:
        return 1
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    try:
        if args.command == "diff":
            return _run_diff(args)
        return asyncio.run(_arun(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
