"""Simulated air traffic density field for congestion-aware routing."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from src.data.ingest import AIRPORTS
from src.utils.geo import great_circle_path


@dataclass
class TrafficField:
    """2D grid of air traffic density (aircraft per grid cell per hour)."""

    lat_grid: np.ndarray
    lon_grid: np.ndarray
    density: np.ndarray
    corridors: list[dict]

    def sample(self, lat: float, lon: float) -> dict:
        """Nearest-neighbor lookup of traffic density at a point."""
        i = int(np.argmin(np.abs(self.lat_grid - lat)))
        j = int(np.argmin(np.abs(self.lon_grid - lon)))
        return {
            "density": float(self.density[i, j]),
            "congestion": float(min(self.density[i, j] / 15.0, 1.0)),
        }


def generate_traffic_field(
    lat_range: tuple[float, float] = (-80, 80),
    lon_range: tuple[float, float] = (-180, 180),
    resolution: float = 2.0,
    seed: int = 42,
) -> TrafficField:
    """Generate a simulated global air traffic density field.

    Models traffic as concentrated along major air corridors between hub
    airports, with Gaussian spread around each corridor centerline.
    """
    rng = np.random.default_rng(seed)
    lat_grid = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lon_grid = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    H, W = len(lat_grid), len(lon_grid)

    density = rng.random((H, W)) * 0.3

    hub_pairs = [
        ("JFK", "LHR", 18.0), ("JFK", "CDG", 12.0), ("JFK", "FRA", 10.0),
        ("LHR", "DXB", 14.0), ("LHR", "SIN", 8.0), ("LHR", "HND", 7.0),
        ("LAX", "HND", 11.0), ("LAX", "SIN", 6.0), ("SFO", "HND", 9.0),
        ("CDG", "DXB", 10.0), ("FRA", "SIN", 8.0), ("DXB", "SIN", 12.0),
        ("ORD", "LHR", 9.0), ("ORD", "FRA", 7.0), ("JFK", "LAX", 16.0),
        ("SFO", "JFK", 14.0), ("LHR", "FRA", 15.0), ("CDG", "FRA", 13.0),
        ("DXB", "HND", 6.0), ("SIN", "HND", 10.0),
    ]

    corridors = []
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing="ij")

    for origin_code, dest_code, intensity in hub_pairs:
        o_lat, o_lon = AIRPORTS[origin_code]
        d_lat, d_lon = AIRPORTS[dest_code]
        path = great_circle_path(o_lat, o_lon, d_lat, d_lon, n_points=40)

        corridor_width_deg = rng.uniform(2.0, 4.0)

        for lat_pt, lon_pt in path:
            dlat = lat_2d - lat_pt
            dlon_raw = lon_2d - lon_pt
            dlon = np.where(dlon_raw > 180, dlon_raw - 360,
                            np.where(dlon_raw < -180, dlon_raw + 360, dlon_raw))
            dist_sq = dlat ** 2 + dlon ** 2
            density += intensity * np.exp(-dist_sq / (2 * corridor_width_deg ** 2))

        corridors.append({
            "origin": origin_code, "destination": dest_code,
            "intensity": intensity,
            "path": path,
        })

    density = gaussian_filter(density, sigma=1.5)
    density = np.clip(density, 0, None)

    return TrafficField(
        lat_grid=lat_grid, lon_grid=lon_grid,
        density=density, corridors=corridors,
    )
