"""Geographic utility functions: haversine, bearing, great-circle interpolation."""

import math
import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing in degrees from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def great_circle_path(
    lat1: float, lon1: float, lat2: float, lon2: float, n_points: int = 50
) -> list[tuple[float, float]]:
    """Interpolate n_points along the great-circle arc between two points."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    d = haversine(lat1, lon1, lat2, lon2) / EARTH_RADIUS_KM

    if d < 1e-12:
        return [(lat1, lon1)] * n_points

    points = []
    for i in range(n_points):
        f = i / max(n_points - 1, 1)
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(lat1r) * math.cos(lon1r) + b * math.cos(lat2r) * math.cos(lon2r)
        y = a * math.cos(lat1r) * math.sin(lon1r) + b * math.cos(lat2r) * math.sin(lon2r)
        z = a * math.sin(lat1r) + b * math.sin(lat2r)
        lat = math.degrees(math.atan2(z, math.sqrt(x ** 2 + y ** 2)))
        lon = math.degrees(math.atan2(y, x))
        points.append((lat, lon))
    return points


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Return (lat, lon) after travelling distance_km along bearing_deg from a starting point."""
    latr = math.radians(lat)
    lonr = math.radians(lon)
    br = math.radians(bearing_deg)
    d = distance_km / EARTH_RADIUS_KM

    lat2 = math.asin(math.sin(latr) * math.cos(d) + math.cos(latr) * math.sin(d) * math.cos(br))
    lon2 = lonr + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(latr),
        math.cos(d) - math.sin(latr) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)
