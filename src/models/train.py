"""Unified training entrypoint for fuel and delay models."""

from pathlib import Path

import pandas as pd
import yaml

from src.models.fuel_predictor import FuelPredictor
from src.models.delay_predictor import DelayPredictor


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def train_all(config: dict | None = None) -> dict:
    """Train both models and save artifacts. Returns metrics for each model."""
    if config is None:
        config = load_config()

    project_root = Path(__file__).resolve().parents[2]
    sample_path = project_root / config["paths"]["sample_data"] / "synthetic_flights.parquet"
    model_dir = project_root / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sample_path)
    results = {}

    print("Training fuel predictor (XGBoost)...")
    fp = FuelPredictor(**config["fuel_model"].get("params", {}))
    fuel_metrics = fp.fit(df)
    fp.save(model_dir / "fuel_predictor.joblib")
    results["fuel"] = fuel_metrics
    print(f"  R²={fuel_metrics['r2']:.4f}  RMSE={fuel_metrics['rmse']:.1f}  "
          f"MAE={fuel_metrics['mae']:.1f}  MAPE={fuel_metrics['mape']:.2f}%")

    print("Training delay predictor (LSTM)...")
    dp = DelayPredictor(**{
        "sequence_length": config["delay_model"]["sequence_length"],
        "hidden_size": config["delay_model"]["hidden_size"],
        "num_layers": config["delay_model"]["num_layers"],
        "dropout": config["delay_model"]["dropout"],
        "epochs": config["delay_model"]["epochs"],
        "batch_size": config["delay_model"]["batch_size"],
        "lr": config["delay_model"]["learning_rate"],
    })
    delay_metrics = dp.fit(df)
    dp.save(model_dir / "delay_predictor.pt")
    results["delay"] = delay_metrics
    print(f"  MAE={delay_metrics['mae']:.2f} min  RMSE={delay_metrics['rmse']:.2f} min")

    return results
