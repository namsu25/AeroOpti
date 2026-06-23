"""Airport and route database from OurAirports and OpenFlights open datasets."""

import csv
import io
import json
from pathlib import Path

import pandas as pd
import requests

_OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
_OPENFLIGHTS_ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
_OPENFLIGHTS_AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _ensure_cache():
    """Create the cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ourairports(use_cache: bool = True) -> pd.DataFrame:
    """Download the OurAirports dataset (large/medium airports with IATA codes).

    Returns DataFrame with columns: iata, name, lat, lon, elevation_ft,
    type, municipality, country_code.
    """
    _ensure_cache()
    cache_path = _CACHE_DIR / "ourairports.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(_OURAIRPORTS_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        df = df[df["iata_code"].notna() & (df["iata_code"] != "")]
        df = df[df["type"].isin(["large_airport", "medium_airport"])]

        result = df.rename(columns={
            "iata_code": "iata",
            "latitude_deg": "lat",
            "longitude_deg": "lon",
        })[["iata", "name", "lat", "lon", "elevation_ft", "type", "municipality", "iso_country"]].copy()
        result = result.rename(columns={"iso_country": "country_code"})
        result = result.drop_duplicates(subset="iata", keep="first").reset_index(drop=True)

        result.to_parquet(cache_path, index=False)
        return result
    except requests.RequestException:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()


def fetch_openflights_routes(use_cache: bool = True) -> pd.DataFrame:
    """Download OpenFlights route database.

    Returns DataFrame with columns: airline, src_iata, dst_iata, codeshare, stops, equipment.
    """
    _ensure_cache()
    cache_path = _CACHE_DIR / "openflights_routes.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(_OPENFLIGHTS_ROUTES_URL, timeout=20)
        resp.raise_for_status()
        cols = ["airline", "airline_id", "src_iata", "src_id", "dst_iata", "dst_id",
                "codeshare", "stops", "equipment"]
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=cols, na_values="\\N")
        df = df[df["src_iata"].notna() & df["dst_iata"].notna()]
        result = df[["airline", "src_iata", "dst_iata", "codeshare", "stops", "equipment"]].copy()
        result.to_parquet(cache_path, index=False)
        return result
    except requests.RequestException:
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()


def get_airport_coords(iata_code: str, airports_df: pd.DataFrame | None = None) -> tuple[float, float] | None:
    """Look up (lat, lon) for an IATA code from the OurAirports dataset."""
    if airports_df is None:
        airports_df = fetch_ourairports()
    row = airports_df[airports_df["iata"] == iata_code]
    if row.empty:
        return None
    return (float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"]))


def get_route_count(src: str, dst: str, routes_df: pd.DataFrame | None = None) -> int:
    """Count how many airlines operate a given route pair."""
    if routes_df is None:
        routes_df = fetch_openflights_routes()
    return len(routes_df[(routes_df["src_iata"] == src) & (routes_df["dst_iata"] == dst)])


def build_airports_dict(codes: list[str], airports_df: pd.DataFrame | None = None) -> dict:
    """Build a {code: [lat, lon]} dict for a list of IATA codes."""
    if airports_df is None:
        airports_df = fetch_ourairports()
    result = {}
    for code in codes:
        coords = get_airport_coords(code, airports_df)
        if coords:
            result[code] = list(coords)
    return result
