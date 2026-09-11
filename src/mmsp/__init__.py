"""
mmsp — multimodal-sports-prediction.

Compares a calibrated sklearn baseline against a multimodal LSTM
(tabular stats + text embeddings) for Premier League match outcome
prediction, validated economically via value betting and fractional
Kelly criterion.

Both pipelines consume the same canonical dataset (mmsp.data) and the
same temporal split (config.yaml) — that is the whole point of the
comparison.
"""

__version__ = "0.1.0"
