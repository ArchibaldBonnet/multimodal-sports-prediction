# multimodal-sports-prediction

Calibrated sklearn baseline vs. a multimodal LSTM (tabular stats + text embeddings) for Premier League match outcome prediction, validated economically via value betting and fractional Kelly criterion[cite: 1].

Both pipelines consume the **same** canonical dataset and the **same** temporal split (`config.yaml`) — that identical footing is what makes the comparison meaningful[cite: 1]. See `src/mmsp/keys.py` for the canonical match key and `src/mmsp/nlp/__init__.py` for the embedding types[cite: 1].

---

## Status

Pipeline fully completed and validated from Phase 0 to Phase 8[cite: 1].

| Phase | Deliverable | Status |
| ----- | ----------- | ------ |
| 0 | Skeleton: repo installs, `python main.py --help` works[cite: 1] | ✅ |
| 1 | Historical dataset consolidated via `football-data.co.uk` CSVs (`E0.csv`)[cite: 1] | ✅ |
| 2 | Tabular features joined by match_id (Shots, Corners, Points, rolling averages)[cite: 1] | ✅ |
| 3 | `dataset.parquet`, one row per match, no silent NaNs[cite: 1] | ✅ |
| 4 | Sklearn baseline, log-loss on real temporal split[cite: 1] | ✅ |
| 5 | Multimodal LSTM (768-dim Guardian NLP + 12-dim Stats) with early stopping & regularization[cite: 1] | ✅ |
| 6 | Log-loss comparison: sklearn vs LSTM (stats-only) vs LSTM (multimodal)[cite: 1] | ✅ |
| 7 | Bankroll curve from real `predict_proba()` / `forward()` calls via Fractional Kelly[cite: 1] | ✅ |
| 8 | Full README, all results reproducible in one command (`python main.py all`)[cite: 1] | ✅ |

---

## Installation

```bash
git clone [https://github.com/ArchibaldBonnet/multimodal-sports-prediction.git](https://github.com/ArchibaldBonnet/multimodal-sports-prediction.git)
cd multimodal-sports-prediction
pip install -e .