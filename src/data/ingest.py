"""Data ingestion: OpenSky live API, Kaggle CSV loader, and synthetic flight generator."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AIRCRAFT_SPECS = {
    "A320": {"fuel_rate": 2.55, "payload_kg": 16600, "cruise_alt_ft": 36000},
    "A330": {"fuel_rate": 5.30, "payload_kg": 47900, "cruise_alt_ft": 37000},
    "A350": {"fuel_rate": 5.80, "payload_kg": 53300, "cruise_alt_ft": 39000},
    "A380": {"fuel_rate": 11.90, "payload_kg": 84000, "cruise_alt_ft": 38000},
    "B737": {"fuel_rate": 2.65, "payload_kg": 18700, "cruise_alt_ft": 36000},
    "B747": {"fuel_rate": 10.10, "payload_kg": 76000, "cruise_alt_ft": 35000},
    "B777": {"fuel_rate": 6.85, "payload_kg": 64500, "cruise_alt_ft": 37000},
    "B787": {"fuel_rate": 5.40, "payload_kg": 51000, "cruise_alt_ft": 39000},
}

NARROW_BODY = {"A320", "B737"}
WIDE_BODY = {"A350", "B777", "B787", "A380"}
MID_BODY = {"A330", "B747"}

AIRPORTS = {
    "JFK": (40.6413, -73.7781), "LHR": (51.4700, -0.4543),
    "LAX": (33.9416, -118.4085), "CDG": (49.0097, 2.5479),
    "DXB": (25.2532, 55.3657), "SIN": (1.3644, 103.9915),
    "HND": (35.5494, 139.7798), "FRA": (50.0379, 8.5622),
    "ORD": (41.9742, -87.9073), "SFO": (37.6213, -122.3790),
}


def fetch_opensky_states() -> pd.DataFrame:
    """Fetch live state vectors from the OpenSky Network REST API."""
    try:
        resp = requests.get("https://opensky-network.org/api/states/all", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        cols = ["icao24", "callsign", "origin_country", "time_position", "last_contact",
                "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
                "true_track", "vertical_rate", "sensors", "geo_altitude",
                "squawk", "spi", "position_source"]
        df = pd.DataFrame(data["states"], columns=cols)
        return df[["icao24", "latitude", "longitude", "baro_altitude", "velocity", "true_track"]]
    except (requests.RequestException, KeyError, json.JSONDecodeError):
        return pd.DataFrame()


def load_kaggle_csv(path: str | Path) -> pd.DataFrame:
    """Load a Kaggle flight-delay CSV, selecting known columns that exist."""
    known_cols = [
        "flight_date", "airline", "origin", "destination", "scheduled_departure",
        "actual_departure", "departure_delay", "arrival_delay", "distance", "air_time",
    ]
    df = pd.read_csv(path)
    present = [c for c in known_cols if c in df.columns]
    return df[present]


def _haversine_vec(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def generate_synthetic_flights(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate n realistic synthetic flight records with correlated features."""
    rng = np.random.default_rng(seed)
    codes = list(AIRPORTS.keys())
    pairs = [(o, d) for o, d in itertools.product(codes, repeat=2) if o != d]

    records = []
    for _ in range(n):
        origin, dest = pairs[rng.integers(len(pairs))]
        olat, olon = AIRPORTS[origin]
        dlat, dlon = AIRPORTS[dest]
        distance = float(_haversine_vec(
            np.array([olat]), np.array([olon]), np.array([dlat]), np.array([dlon])
        )[0])

        if distance < 2000:
            ac = rng.choice(list(NARROW_BODY))
        elif distance > 6000:
            ac = rng.choice(list(WIDE_BODY))
        else:
            ac = rng.choice(list(MID_BODY | WIDE_BODY))

        spec = AIRCRAFT_SPECS[ac]
        payload_factor = rng.uniform(0.5, 1.0)
        payload_kg = spec["payload_kg"] * payload_factor
        headwind_kts = rng.normal(15, 20)
        headwind_ms = headwind_kts * 0.5144
        temp_dev_c = rng.normal(0, 4)
        turbulence_idx = rng.beta(2, 8)

        wind_factor = headwind_ms / 250
        temp_factor = temp_dev_c / 100
        turb_detour = turbulence_idx * 0.03
        noise = rng.normal(0, 0.02)

        fuel_kg = spec["fuel_rate"] * distance * (
            1 + (payload_factor - 0.75) * 0.15 + wind_factor + temp_factor + turb_detour + noise
        )
        fuel_kg = max(fuel_kg, distance * spec["fuel_rate"] * 0.7)

        ground_speed = 850 - headwind_ms * 1.852
        flight_time_h = distance / max(ground_speed, 400) + 0.5

        dep_delay = max(0, rng.gamma(2, 4) + turbulence_idx * rng.uniform(0, 30))
        arr_delay = dep_delay + rng.normal(0, 5) + turbulence_idx * 10
        hour = rng.integers(0, 24)
        day_of_week = rng.integers(0, 7)
        flight_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))

        records.append({
            "flight_date": flight_date,
            "origin": origin,
            "destination": dest,
            "aircraft_type": ac,
            "distance_km": round(distance, 1),
            "cruise_alt_ft": spec["cruise_alt_ft"],
            "payload_kg": round(payload_kg, 1),
            "headwind_kts": round(headwind_kts, 1),
            "temp_dev_c": round(temp_dev_c, 2),
            "turbulence_idx": round(turbulence_idx, 4),
            "fuel_kg": round(fuel_kg, 1),
            "flight_time_h": round(flight_time_h, 2),
            "departure_delay_min": round(dep_delay, 1),
            "arrival_delay_min": round(arr_delay, 1),
            "hour": hour,
            "day_of_week": day_of_week,
        })

    return pd.DataFrame(records)
