"""CLI entrypoint to train both ML models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import load_config, train_all


def main():
    """Parse args and train all models."""
    parser = argparse.ArgumentParser(description="Train AeroOpti ML models")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    results = train_all(config)

    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    print("\nFuel Predictor (XGBoost):")
    for k, v in results["fuel"].items():
        print(f"  {k}: {v:.4f}")
    print("\nDelay Predictor (LSTM):")
    for k, v in results["delay"].items():
        print(f"  {k}: {v:.4f}")

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / config["paths"]["models"]
    print(f"\nArtifacts saved to: {model_dir}")
    print("  - fuel_predictor.xgb.json / fuel_predictor.meta.json")
    print("  - delay_predictor.pt")
    print("  - latest_metrics.json / training_history.jsonl")


if __name__ == "__main__":
    main()
