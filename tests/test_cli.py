"""Tests for the diff CLI subcommand argument handling."""

from harness.cli import build_parser


def test_diff_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["diff", "base.json", "cand.json"])
    assert args.command == "diff"
    assert args.baseline == "base.json"
    assert args.candidate == "cand.json"


def test_run_subcommand_still_works():
    parser = build_parser()
    args = parser.parse_args(["run", "evals/x.yaml", "--model", "m"])
    assert args.command == "run"
    assert args.eval_file == "evals/x.yaml"
    assert args.model == "m"


def test_diff_show_all_flag():
    parser = build_parser()
    args = parser.parse_args(["diff", "a.json", "b.json", "--show-all"])
    assert args.show_all is True


def test_diff_json_out_flag():
    parser = build_parser()
    args = parser.parse_args(["diff", "a.json", "b.json", "--json", "out.json"])
    assert args.json_out == "out.json"


def test_diff_fail_on_regression_default_true():
    parser = build_parser()
    args = parser.parse_args(["diff", "a.json", "b.json"])
    assert args.fail_on_regression is True


def test_diff_no_fail_flag():
    parser = build_parser()
    args = parser.parse_args(["diff", "a.json", "b.json", "--no-fail-on-regression"])
    assert args.fail_on_regression is False
