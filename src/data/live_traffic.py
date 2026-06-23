"""Live flight traffic from OpenSky Network REST API."""

import json

import numpy as np
import pandas as pd
import requests


def fetch_opensky_live(
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Fetch current airborne aircraft from OpenSky Network.

    bbox = (min_lat, min_lon, max_lat, max_lon) or None for global.
    No API key needed for anonymous access (rate-limited to ~100 req/day).

    Returns DataFrame: icao24, callsign, origin_country, lat, lon,
    altitude_m, velocity_ms, track_deg, vertical_rate, on_ground.
    """
    try:
        url = "https://opensky-network.org/api/states/all"
        params = {}
        if bbox:
            params.update({
                "lamin": bbox[0], "lomin": bbox[1],
                "lamax": bbox[2], "lomax": bbox[3],
            })

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("states"):
            return pd.DataFrame()

        cols = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi", "position_source",
        ]
        df = pd.DataFrame(data["states"], columns=cols)

        df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
        df["callsign"] = df["callsign"].str.strip()

        return df.rename(columns={
            "latitude": "lat",
            "longitude": "lon",
            "baro_altitude": "altitude_m",
            "velocity": "velocity_ms",
            "true_track": "track_deg",
        })[["icao24", "callsign", "origin_country", "lat", "lon",
            "altitude_m", "velocity_ms", "track_deg", "vertical_rate", "on_ground"]].reset_index(drop=True)

    except (requests.RequestException, KeyError, json.JSONDecodeError):
        return pd.DataFrame()


def fetch_opensky_bbox(
    origin: tuple[float, float],
    destination: tuple[float, float],
    margin_deg: float = 5.0,
) -> pd.DataFrame:
    """Fetch live traffic in the bounding box of a route with margin."""
    min_lat = min(origin[0], destination[0]) - margin_deg
    max_lat = max(origin[0], destination[0]) + margin_deg
    min_lon = min(origin[1], destination[1]) - margin_deg
    max_lon = max(origin[1], destination[1]) + margin_deg
    return fetch_opensky_live(bbox=(min_lat, min_lon, max_lat, max_lon))


def traffic_density_from_live(
    df: pd.DataFrame,
    lat_range: tuple[float, float] = (-80, 80),
    lon_range: tuple[float, float] = (-180, 180),
    resolution: float = 2.0,
) -> dict:
    """Convert live aircraft positions into a density grid.

    Returns dict with lat_grid, lon_grid, density (2D array of aircraft count per cell).
    """
    lat_grid = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon_grid = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    density = np.zeros((len(lat_grid), len(lon_grid)))

    if df.empty:
        return {"lat_grid": lat_grid, "lon_grid": lon_grid, "density": density}

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        i = int(np.argmin(np.abs(lat_grid - lat)))
        j = int(np.argmin(np.abs(lon_grid - lon)))
        density[i, j] += 1

    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=1.0)
    return {"lat_grid": lat_grid, "lon_grid": lon_grid, "density": density}
