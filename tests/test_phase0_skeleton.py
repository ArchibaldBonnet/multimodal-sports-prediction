"""
Phase 0 gate: "repo installs; python main.py --help works".

These tests check the skeleton only — no data, no models. They exist
so Phase 1 starts from a verified baseline instead of an assumption.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import mmsp
    assert mmsp.__version__ == "0.1.0"


def test_canonical_key_contract_reserved():
    # Phase 1 implements the bodies; Phase 0 only reserves the contract
    # so nothing downstream can invent its own ad hoc join logic instead.
    from mmsp import keys
    assert hasattr(keys, "normalize_team")
    assert hasattr(keys, "match_id")


def test_all_subpackages_importable():
    import mmsp.data
    import mmsp.nlp
    import mmsp.features
    import mmsp.models
    import mmsp.models.baseline
    import mmsp.models.lstm
    import mmsp.evaluation
    import mmsp.backtesting


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    for cmd in ["fetch", "build-features", "train-baseline", "train-lstm", "evaluate", "backtest"]:
        assert cmd in result.stdout


def test_cli_subcommand_fails_loudly_not_silently():
    # Design principle: unimplemented steps must fail with a clear
    # phase-tagged error, never silently return / print fake output.
    result = subprocess.run(
        [sys.executable, "main.py", "fetch"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "not yet implemented" in result.stderr
    assert "Phase 1" in result.stderr
