"""
mmsp.models.baseline — calibrated sklearn baseline (Phase 4).

Trained and evaluated on the flat canonical dataset, using the
temporal split from config.yaml. Calibration (isotonic/Platt) is
required: uncalibrated probabilities are not valid inputs to Kelly.
"""
