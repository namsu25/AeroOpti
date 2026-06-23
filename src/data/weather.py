"""Weather field simulator: turbulence risk, wind, and temperature grids."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class WeatherField:
    """2D weather grid with risk, wind components, and temperature deviation."""

    lat_grid: np.ndarray
    lon_grid: np.ndarray
    risk: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    temperature: np.ndarray

    def sample(self, lat: float, lon: float) -> dict:
        """Nearest-neighbor lookup of weather at a point."""
        i = int(np.argmin(np.abs(self.lat_grid - lat)))
        j = int(np.argmin(np.abs(self.lon_grid - lon)))
        return {
            "risk": float(self.risk[i, j]),
            "wind_u": float(self.wind_u[i, j]),
            "wind_v": float(self.wind_v[i, j]),
            "temp_dev_c": float(self.temperature[i, j]),
        }

    def headwind_along(self, lat: float, lon: float, bearing_deg: float) -> float:
        """Positive return = headwind in m/s along bearing_deg."""
        wx = self.sample(lat, lon)
        br = math.radians(bearing_deg)
        flight_u = math.sin(br)
        flight_v = math.cos(br)
        return -(wx["wind_u"] * flight_u + wx["wind_v"] * flight_v)


def generate_weather_field(
    lat_range: tuple[float, float] = (-80, 80),
    lon_range: tuple[float, float] = (-180, 180),
    resolution: float = 2.0,
    n_storms: int = 8,
    seed: int = 42,
) -> WeatherField:
    """Generate a plausible global weather field."""
    rng = np.random.default_rng(seed)
    lat_grid = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon_grid = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    H, W = len(lat_grid), len(lon_grid)

    risk = rng.random((H, W)) * 0.15
    for _ in range(n_storms):
        ci, cj = rng.integers(0, H), rng.integers(0, W)
        intensity = rng.uniform(0.6, 1.0)
        radius = rng.integers(4, 12)
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dist_sq = (ii - ci) ** 2 + (jj - cj) ** 2
        risk += intensity * np.exp(-dist_sq / (2 * radius ** 2))
    risk = gaussian_filter(risk, sigma=2)
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-12)

    lat_2d = np.tile(lat_grid[:, None], (1, W))
    jet_profile = np.exp(-((lat_2d - 35) ** 2) / (2 * 20 ** 2))
    wind_u = jet_profile * 25 + rng.normal(0, 4, (H, W))
    wind_u = gaussian_filter(wind_u, sigma=1.5)
    wind_v = rng.normal(0, 3, (H, W))
    wind_v = gaussian_filter(wind_v, sigma=1.5)

    temperature = rng.normal(0, 4, (H, W))
    temperature = gaussian_filter(temperature, sigma=2)

    return WeatherField(
        lat_grid=lat_grid, lon_grid=lon_grid,
        risk=risk, wind_u=wind_u, wind_v=wind_v, temperature=temperature,
    )
