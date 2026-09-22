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
import os
from dotenv import load_dotenv
from src.mmsp.fetch import fetch_premier_league_data
from src.mmsp.data.build import build_canonical_dataset
from src.mmsp.models.evaluate import get_prepared_data
from src.mmsp.models.baseline import train_sklearn_baseline
from src.mmsp.models.lstm import train_multimodal_lstm
from src.mmsp.models.evaluate import run_evaluation
from src.mmsp.models.backtesting import run_financial_backtest
from src.mmsp.models.ablation import run_embedding_ablation
from src.mmsp.models.analyze import generate_betting_report
import numpy as np
import random
import torch

def set_seed(seed=42):
    """Fixe l'aléatoire pour garantir des bankrolls identiques."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Verrouillage exécuté au lancement du script
set_seed(42)

def cmd_fetch(args):
    load_dotenv()  # Charge votre .env contenant FOOTBALL_DATA_API_KEY
    print("Démarrage de la Phase 1 : Récupération des données...")
    fetch_premier_league_data()


def cmd_build_features(args):
    print("Lancement du nouveau pipeline CSV...")
    build_canonical_dataset()


def cmd_train_baseline(args):
    print("Démarrage de la Phase 4 : Entraînement de la Sklearn Baseline...")
    train_sklearn_baseline()


def cmd_train_lstm(args):
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = get_prepared_data()
    train_multimodal_lstm(X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test, use_text=True)


def cmd_evaluate(args):
    run_evaluation()


def cmd_backtest(args):
    run_financial_backtest()

def cmd_ablation(args):
    run_embedding_ablation()

def cmd_analyze(args):
    generate_betting_report()


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

    p = sub.add_parser("ablation", help="Study NLP impact")
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("analyze", help="CSV bets analysis of the 3 models")
    p.set_defaults(func=cmd_analyze)

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
