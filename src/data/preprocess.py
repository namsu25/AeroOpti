"""Feature engineering for fuel and delay prediction models."""

import numpy as np
import pandas as pd


def build_fuel_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target vector for fuel prediction."""
    ac_dummies = pd.get_dummies(df["aircraft_type"], prefix="ac", dtype=float)
    X = pd.concat([
        df[["distance_km", "cruise_alt_ft", "payload_kg",
            "headwind_kts", "temp_dev_c", "turbulence_idx"]].reset_index(drop=True),
        ac_dummies.reset_index(drop=True),
    ], axis=1)
    y = df["fuel_kg"].reset_index(drop=True)
    return X, y


def build_delay_sequences(
    df: pd.DataFrame, sequence_length: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for LSTM delay prediction."""
    feature_cols = [
        "departure_delay_min", "arrival_delay_min",
        "turbulence_idx", "headwind_kts", "hour", "day_of_week",
    ]
    df = df.sort_values("flight_date").reset_index(drop=True)
    sequences, targets = [], []

    for _, group in df.groupby(["origin", "destination"]):
        if len(group) < sequence_length + 1:
            continue
        vals = group[feature_cols].values
        target_vals = group["arrival_delay_min"].values
        for i in range(len(vals) - sequence_length):
            sequences.append(vals[i : i + sequence_length])
            targets.append(target_vals[i + sequence_length])

    if not sequences:
        return np.empty((0, sequence_length, len(feature_cols))), np.empty((0,))
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def normalize_sequences(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize per channel across all samples. Returns (X_norm, mean, std)."""
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1)) + 1e-8
    X_norm = (X - mean) / std
    return X_norm, mean, std
