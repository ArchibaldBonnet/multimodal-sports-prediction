"""
mmsp.models.lstm — multimodal LSTM (Phase 5).

Sequences are reconstructed per team, sorted by date, joined on
match_id — not by row position in a CSV. Consumes the same
dataset.parquet as the baseline, restricted to the same temporal
split. Real per-epoch loss must be logged to CSV during training
(no handwritten/fabricated curves).
"""
