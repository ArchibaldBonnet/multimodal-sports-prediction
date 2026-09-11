"""
mmsp.models — the two compared pipelines.

- mmsp.models.baseline: calibrated sklearn classifier, flat features
  from dataset.parquet (Phase 4).
- mmsp.models.lstm: multimodal LSTM, sequences reconstructed per team
  from the SAME dataset.parquet, joined on match_id (Phase 5).

Both consume mmsp.data's canonical dataset and the split defined once
in config.yaml — never a separate split computed locally.
"""
