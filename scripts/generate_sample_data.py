"""Generate synthetic flight dataset for offline development."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import generate_synthetic_flights, AIRPORTS


def main():
    """Generate and save synthetic flights + airports JSON."""
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 10,000 synthetic flights (seed=42)...")
    df = generate_synthetic_flights(n=10000, seed=42)

    parquet_path = sample_dir / "synthetic_flights.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved to {parquet_path}")

    airports_path = sample_dir / "airports.json"
    with open(airports_path, "w") as f:
        json.dump({k: list(v) for k, v in AIRPORTS.items()}, f, indent=2)
    print(f"Saved airports to {airports_path}")

    print(f"\nSummary:")
    print(f"  Rows: {len(df)}")
    print(f"  Distance range: {df['distance_km'].min():.0f} – {df['distance_km'].max():.0f} km")
    print(f"  Fuel range: {df['fuel_kg'].min():.0f} – {df['fuel_kg'].max():.0f} kg")
    print(f"  Aircraft types: {sorted(df['aircraft_type'].unique())}")
    print(f"  Routes: {df.groupby(['origin','destination']).ngroups} unique pairs")
    print(f"\nGenerated 10000 flights, saved to data/sample/")


if __name__ == "__main__":
    main()
