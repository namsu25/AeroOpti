"""Tests for ML model wrappers."""

import tempfile
from pathlib import Path

import numpy as np

from src.data.ingest import generate_synthetic_flights
from src.data.preprocess import build_delay_sequences
from src.models.fuel_predictor import FuelPredictor


def test_fuel_predictor_fit_predict():
    """Fuel predictor should achieve R² > 0.7 on 200 rows."""
    df = generate_synthetic_flights(n=200, seed=123)
    fp = FuelPredictor(n_estimators=50, max_depth=4)
    metrics = fp.fit(df)
    assert metrics["r2"] > 0.7
    pred = fp.predict_row("B777", 5500, 37000, 48000, 15, 0.5, 0.1)
    assert pred > 0


def test_fuel_predictor_save_load():
    """Save and load should produce identical predictions."""
    df = generate_synthetic_flights(n=200, seed=123)
    fp = FuelPredictor(n_estimators=50, max_depth=4)
    fp.fit(df)
    pred_before = fp.predict_row("B777", 5500, 37000, 48000, 15, 0.5, 0.1)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_model.joblib"
        fp.save(path)
        fp2 = FuelPredictor.load(path)
        pred_after = fp2.predict_row("B777", 5500, 37000, 48000, 15, 0.5, 0.1)

    assert abs(pred_before - pred_after) / pred_before < 0.001


def test_delay_sequences_shape():
    """Delay sequence builder should produce correct tensor shapes."""
    df = generate_synthetic_flights(n=500, seed=42)
    X, y = build_delay_sequences(df, sequence_length=10)
    assert X.ndim == 3
    assert X.shape[1] == 10
    assert X.shape[2] == 6
    assert len(y) == len(X)
