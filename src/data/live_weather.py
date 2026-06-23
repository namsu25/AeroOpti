"""Live weather data from Open-Meteo and NOAA Aviation Weather APIs."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter

from src.data.weather import WeatherField


def fetch_openmeteo_winds(
    lat_range: tuple[float, float] = (20, 65),
    lon_range: tuple[float, float] = (-90, 20),
    resolution: float = 2.0,
) -> WeatherField | None:
    """Fetch upper-level winds and temperature from Open-Meteo forecast API.

    Queries the 250hPa pressure level (approx FL340) for wind_u, wind_v, and
    temperature across a lat/lon grid. Falls back to None on failure.
    """
    lat_grid = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon_grid = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    H, W = len(lat_grid), len(lon_grid)

    wind_u = np.zeros((H, W))
    wind_v = np.zeros((H, W))
    temperature = np.zeros((H, W))
    risk = np.zeros((H, W))

    sample_lats = lat_grid[::max(1, H // 8)]
    sample_lons = lon_grid[::max(1, W // 8)]

    for lat in sample_lats:
        for lon in sample_lons:
            try:
                params = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "hourly": "wind_speed_250hPa,wind_direction_250hPa,temperature_250hPa",
                    "forecast_days": 1,
                    "timezone": "UTC",
                }
                resp = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params=params, timeout=8,
                )
                resp.raise_for_status()
                data = resp.json()
                hourly = data.get("hourly", {})

                ws_list = hourly.get("wind_speed_250hPa", [])
                wd_list = hourly.get("wind_direction_250hPa", [])
                t_list = hourly.get("temperature_250hPa", [])

                if not ws_list or not wd_list:
                    continue

                ws = _first_valid(ws_list)
                wd = _first_valid(wd_list)
                t = _first_valid(t_list) if t_list else 0.0

                if ws is None or wd is None:
                    continue

                ws_ms = ws / 3.6
                wd_rad = math.radians(wd)
                u = -ws_ms * math.sin(wd_rad)
                v = -ws_ms * math.cos(wd_rad)

                i = int(np.argmin(np.abs(lat_grid - lat)))
                j = int(np.argmin(np.abs(lon_grid - lon)))
                wind_u[i, j] = u
                wind_v[i, j] = v
                temperature[i, j] = t + 56.5

            except (requests.RequestException, KeyError, ValueError):
                continue

    wind_u = gaussian_filter(wind_u, sigma=2)
    wind_v = gaussian_filter(wind_v, sigma=2)
    temperature = gaussian_filter(temperature, sigma=2)

    speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
    wind_shear = np.gradient(speed, axis=0) ** 2 + np.gradient(speed, axis=1) ** 2
    risk = np.sqrt(wind_shear)
    risk = risk / (risk.max() + 1e-12)
    risk = gaussian_filter(risk, sigma=1.5)

    return WeatherField(
        lat_grid=lat_grid, lon_grid=lon_grid,
        risk=risk, wind_u=wind_u, wind_v=wind_v, temperature=temperature,
    )


@dataclass
class AviationWeatherReport:
    """Parsed NOAA aviation weather report (METAR, SIGMET, or PIREP)."""
    report_type: str
    raw_text: str
    station_id: str = ""
    lat: float = 0.0
    lon: float = 0.0
    flight_category: str = ""
    wind_speed_kts: float = 0.0
    wind_dir_deg: float = 0.0
    visibility_mi: float = 10.0
    ceiling_ft: float = 99999.0
    temp_c: float = 0.0
    altimeter_inhg: float = 29.92
    severity: str = ""


def fetch_noaa_metars(bbox: tuple[float, float, float, float] | None = None) -> list[AviationWeatherReport]:
    """Fetch current METARs from NOAA Aviation Weather Center.

    bbox = (min_lat, min_lon, max_lat, max_lon) or None for global.
    Uses the aviationweather.gov v1 data API.
    """
    try:
        url = "https://aviationweather.gov/api/data/metar"
        params = {"format": "json", "hours": 2}
        if bbox:
            params["bbox"] = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        reports = []
        for m in data:
            reports.append(AviationWeatherReport(
                report_type="METAR",
                raw_text=m.get("rawOb", ""),
                station_id=m.get("icaoId", ""),
                lat=float(m.get("lat", 0)),
                lon=float(m.get("lon", 0)),
                flight_category=m.get("fltcat", ""),
                wind_speed_kts=float(m.get("wspd", 0) or 0),
                wind_dir_deg=float(m.get("wdir", 0) or 0),
                visibility_mi=float(m.get("visib", 10) or 10),
                ceiling_ft=float(m.get("ceil", 99999) or 99999),
                temp_c=float(m.get("temp", 0) or 0),
                altimeter_inhg=float(m.get("altim", 29.92) or 29.92),
            ))
        return reports
    except (requests.RequestException, KeyError, ValueError):
        return []


def fetch_noaa_sigmets() -> list[AviationWeatherReport]:
    """Fetch active SIGMETs/AIRMETs from NOAA Aviation Weather Center."""
    try:
        url = "https://aviationweather.gov/api/data/airsigmet"
        params = {"format": "json"}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        reports = []
        for s in data:
            coords = s.get("coords", [])
            if coords:
                lats = [c.get("lat", 0) for c in coords]
                lons = [c.get("lon", 0) for c in coords]
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)
            else:
                center_lat, center_lon = 0.0, 0.0

            reports.append(AviationWeatherReport(
                report_type=s.get("airsigmetType", "SIGMET"),
                raw_text=s.get("rawAirSigmet", ""),
                lat=center_lat,
                lon=center_lon,
                severity=s.get("severity", ""),
            ))
        return reports
    except (requests.RequestException, KeyError, ValueError):
        return []


def metars_to_dataframe(metars: list[AviationWeatherReport]) -> pd.DataFrame:
    """Convert METAR reports to a DataFrame for display and mapping."""
    if not metars:
        return pd.DataFrame()
    records = []
    for m in metars:
        records.append({
            "station": m.station_id,
            "lat": m.lat,
            "lon": m.lon,
            "category": m.flight_category,
            "wind_kts": m.wind_speed_kts,
            "wind_dir": m.wind_dir_deg,
            "visibility_mi": m.visibility_mi,
            "ceiling_ft": m.ceiling_ft,
            "temp_c": m.temp_c,
            "raw": m.raw_text,
        })
    return pd.DataFrame(records)


def _first_valid(lst: list) -> float | None:
    """Return the first non-None numeric value from a list."""
    for v in lst:
        if v is not None:
            return float(v)
    return None
