"""
mmsp.data — ingestion and canonical dataset construction.

Phase 1: fetch raw matches/odds, assign match_id (mmsp.keys), produce
matches.parquet with a unique match_id per row.

Phase 3: build_dataset.py combines matches, tabular features
(mmsp.features), and embeddings (mmsp.nlp) into the single canonical
dataset.parquet consumed identically by the baseline and the LSTM.
"""
