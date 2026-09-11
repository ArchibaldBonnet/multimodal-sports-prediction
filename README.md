# multimodal-sports-prediction

Calibrated sklearn baseline vs. a multimodal LSTM (tabular stats + text
embeddings) for Premier League match outcome prediction, validated
economically via value betting and fractional Kelly criterion.

Both pipelines consume the **same** canonical dataset and the **same**
temporal split (`config.yaml`) — that identical footing is what makes
the comparison meaningful. See `src/mmsp/keys.py` for the canonical
match key and `src/mmsp/nlp/__init__.py` for the two embedding types.

## Status

Phase 0 (skeleton) — repo installs, `python main.py --help` works, all
subcommands are wired but intentionally raise `NotImplementedError`
until their phase lands.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Skeleton: repo installs, `main.py --help` works | ✅ |
| 1 | `matches.parquet` with unique `match_id`, tests pass | — |
| 2 | Tabular features joined by `match_id` | — |
| 3 | `dataset.parquet`, one row per match, no silent NaNs | — |
| 4 | sklearn baseline, log-loss on real temporal split | — |
| 5 | Multimodal LSTM, real logged loss curves | — |
| 6 | Log-loss comparison: sklearn vs LSTM(stats-only) vs LSTM(multimodal) | — |
| 7 | Bankroll curve from real `predict_proba()`/`forward()` calls | — |
| 8 | Full README, all results reproducible in one command | — |

## Install

```bash
pip install -e .
```

## Usage

```bash
python main.py --help
python main.py fetch              # Phase 1
python main.py build-features     # Phase 2/3
python main.py train-baseline     # Phase 4
python main.py train-lstm         # Phase 5
python main.py evaluate           # Phase 6
python main.py backtest           # Phase 7
```

## Tests

```bash
pytest tests/ -v
```
