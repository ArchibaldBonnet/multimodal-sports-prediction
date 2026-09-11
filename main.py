#!/usr/bin/env python
"""
main.py — unified CLI entry point for multimodal-sports-prediction.

Each subcommand lands in the phase named below. Until then it fails
loudly with NotImplementedError rather than silently doing nothing —
the original codebase "appeared" to run without producing anything
checkable, and that failure mode is exactly what this CLI is meant to
prevent.

    fetch            Phase 1 — download raw matches/odds/news, assign match_id
    build-features   Phase 2/3 — tabular features + embeddings -> dataset.parquet
    train-baseline   Phase 4 — calibrated sklearn baseline
    train-lstm       Phase 5 — multimodal LSTM
    evaluate         Phase 6 — sklearn vs LSTM(stats-only) vs LSTM(multimodal)
    backtest         Phase 7 — value-bet / fractional-Kelly backtest
"""
import argparse
import sys


def cmd_fetch(args):
    raise NotImplementedError("fetch: implemented in Phase 1 (ingestion + canonical key)")


def cmd_build_features(args):
    raise NotImplementedError("build-features: implemented in Phase 2/3 (tabular features + embeddings)")


def cmd_train_baseline(args):
    raise NotImplementedError("train-baseline: implemented in Phase 4 (sklearn baseline)")


def cmd_train_lstm(args):
    raise NotImplementedError("train-lstm: implemented in Phase 5 (multimodal LSTM)")


def cmd_evaluate(args):
    raise NotImplementedError("evaluate: implemented in Phase 6 (real evaluation)")


def cmd_backtest(args):
    raise NotImplementedError("backtest: implemented in Phase 7 (connected backtesting)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Multimodal Premier League match prediction: calibrated sklearn "
            "baseline vs multimodal LSTM, validated via Kelly-criterion backtesting."
        ),
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the canonical config (temporal split, paths, hyperparameters). Default: %(default)s",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="Download raw match/odds/news data")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build-features", help="Build the canonical dataset (tabular features + embeddings)")
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("train-baseline", help="Train the calibrated sklearn baseline")
    p.set_defaults(func=cmd_train_baseline)

    p = sub.add_parser("train-lstm", help="Train the multimodal LSTM")
    p.set_defaults(func=cmd_train_lstm)

    p = sub.add_parser("evaluate", help="Compare sklearn vs LSTM(stats-only) vs LSTM(multimodal) on the test set")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("backtest", help="Run the value-bet / fractional-Kelly backtest")
    p.set_defaults(func=cmd_backtest)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except NotImplementedError as e:
        print(f"[not yet implemented] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
