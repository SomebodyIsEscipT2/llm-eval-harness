"""CLI entrypoint: run an eval file and print a report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from harness.eval_file import load_suite
from harness.providers import available as available_providers
from harness.providers.base import ProviderError
from harness.report import report
from harness.runner import run_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness",
        description="Run structured evaluations against an LLM provider.",
    )
    p.add_argument("eval_file", help="Path to a .json/.yaml eval suite")
    p.add_argument(
        "--provider", default="openrouter", help="Provider name (default: openrouter)"
    )
    p.add_argument("--model", default=None, help="Model override for the provider")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--json", dest="json_out", default=None, help="Write run artifact to this path")
    p.add_argument("--failures-only", action="store_true", help="Show only failing cases")
    return p


async def _amain(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.eval_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from harness.providers import get as get_provider

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
        result = await run_suite(
            suite,
            provider,
            model=args.model,
            concurrency=args.concurrency,
            retries=args.retries,
        )
    finally:
        await provider.aclose()

    print(report(result, only_failures=args.failures_only))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nartifact written to {args.json_out}")

    return 0 if result.passed == result.total else 1


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
