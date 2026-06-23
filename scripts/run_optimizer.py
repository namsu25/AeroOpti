"""CLI entrypoint to run the route optimizer."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.weather import generate_weather_field
from src.data.traffic import generate_traffic_field
from src.models.fuel_predictor import FuelPredictor
from src.models.train import load_config
from src.optimization.optimizer import RouteOptimizer
from src.utils.metrics import route_savings_pct


def main():
    """Run route optimization and print comparison table."""
    parser = argparse.ArgumentParser(description="AeroOpti Route Optimizer")
    parser.add_argument("--origin", default="JFK", help="Origin airport code")
    parser.add_argument("--destination", default="LHR", help="Destination airport code")
    parser.add_argument("--aircraft", default="B777", help="Aircraft type")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / config["paths"]["models"]

    print(f"Loading fuel predictor from {model_dir / 'fuel_predictor.joblib'}...")
    fp = FuelPredictor.load(model_dir / "fuel_predictor.joblib")

    print("Generating weather field...")
    wf = generate_weather_field(seed=config["data"]["random_seed"])

    print("Generating traffic field...")
    tf = generate_traffic_field(seed=config["data"]["random_seed"])

    airports = config["airports"]
    weights = config["optimizer"]["cost_weights"]

    spec_lookup = {
        "A320": 16600, "A330": 47900, "A350": 53300, "A380": 84000,
        "B737": 18700, "B747": 76000, "B777": 64500, "B787": 51000,
    }
    alt_lookup = {
        "A320": 36000, "A330": 37000, "A350": 39000, "A380": 38000,
        "B737": 36000, "B747": 35000, "B777": 37000, "B787": 39000,
    }

    payload_kg = spec_lookup[args.aircraft] * 0.75
    cruise_alt_ft = alt_lookup[args.aircraft]

    print(f"\nOptimizing: {args.origin} -> {args.destination} ({args.aircraft})")
    print(f"  Payload: {payload_kg:.0f} kg | Cruise: FL{cruise_alt_ft // 100}")
    print("-" * 70)

    optimizer = RouteOptimizer(fp, wf, config, traffic_field=tf)
    routes = optimizer.optimize(
        args.origin, args.destination, args.aircraft,
        payload_kg, cruise_alt_ft, weights, airports,
    )

    header = f"{'Route':<25} {'Fuel (kg)':>10} {'Time (h)':>9} {'Risk':>6} {'Traffic':>8} {'CO2 (kg)':>10} {'Cost ($)':>10}"
    print(header)
    print("-" * len(header))
    for r in routes:
        print(f"{r['name']:<25} {r['fuel_kg']:>10,.0f} {r['time_h']:>9.2f} "
              f"{r['risk']:>6.3f} {r['congestion']:>8.3f} {r['co2_kg']:>10,.0f} {r['cost_usd']:>10,.0f}")

    baseline = routes[0]
    best = min(routes[1:], key=lambda r: r["fuel_kg"])
    print(f"\n% Savings vs baseline ({best['name']}):")
    print(f"  Fuel: {route_savings_pct(baseline['fuel_kg'], best['fuel_kg']):.1f}%")
    print(f"  CO2:  {route_savings_pct(baseline['co2_kg'], best['co2_kg']):.1f}%")
    print(f"  Cost: {route_savings_pct(baseline['cost_usd'], best['cost_usd']):.1f}%")


if __name__ == "__main__":
    main()
