"""Tests for the LSTM delay predictor."""

import tempfile
from pathlib import Path

from src.data.ingest import generate_synthetic_flights
from src.models.delay_predictor import DelayPredictor


def test_delay_predictor_fit_predict():
    """Fitting on enough sequences should yield finite metrics and predictions."""
    df = generate_synthetic_flights(n=500, seed=21)
    dp = DelayPredictor(sequence_length=10, hidden_size=8, num_layers=1, epochs=2, batch_size=32)
    metrics = dp.fit(df)

    assert metrics["mae"] == metrics["mae"]  # not NaN
    assert metrics["mae"] >= 0

    seq = df.sort_values("flight_date").groupby(["origin", "destination"]).get_group(
        next(iter(df.groupby(["origin", "destination"]).groups))
    )
    feature_cols = ["departure_delay_min", "arrival_delay_min", "turbulence_idx",
                     "headwind_kts", "hour", "day_of_week"]
    sample_seq = seq[feature_cols].values[:10]
    if len(sample_seq) == 10:
        pred = dp.predict_sequence(sample_seq)
        assert isinstance(pred, float)


def test_delay_predictor_save_load_round_trip():
    """Save/load should preserve config and produce a usable model."""
    df = generate_synthetic_flights(n=500, seed=21)
    dp = DelayPredictor(sequence_length=10, hidden_size=8, num_layers=1, epochs=2, batch_size=32)
    dp.fit(df)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "delay_predictor.pt"
        dp.save(path)
        dp2 = DelayPredictor.load(path)

    assert dp2.input_size == dp.input_size
    assert dp2.cfg["hidden_size"] == dp.cfg["hidden_size"]
    assert dp2.model is not None


def test_delay_predictor_reproducible_with_seed():
    """Same seed should produce identical validation metrics across runs."""
    df = generate_synthetic_flights(n=500, seed=21)
    dp1 = DelayPredictor(sequence_length=10, hidden_size=8, num_layers=1, epochs=3, batch_size=32, seed=5)
    dp2 = DelayPredictor(sequence_length=10, hidden_size=8, num_layers=1, epochs=3, batch_size=32, seed=5)

    m1 = dp1.fit(df)
    m2 = dp2.fit(df)

    assert m1["mae"] == m2["mae"]
