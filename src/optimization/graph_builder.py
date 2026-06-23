"""Build a waypoint graph over a great-circle corridor for route optimization."""

import math

import networkx as nx
import numpy as np

from src.utils.geo import bearing, destination_point, great_circle_path, haversine


def build_waypoint_graph(
    origin: tuple[float, float],
    destination: tuple[float, float],
    weather_field,
    n_lateral: int = 7,
    n_longitudinal: int = 25,
    lateral_spread_deg: float = 6.0,
    traffic_field=None,
) -> nx.DiGraph:
    """Construct a directed graph of waypoints along a great-circle corridor."""
    gc_points = great_circle_path(
        origin[0], origin[1], destination[0], destination[1], n_points=n_longitudinal,
    )

    G = nx.DiGraph()
    source_id = "SRC"
    sink_id = "SNK"
    G.add_node(source_id, lat=origin[0], lon=origin[1])
    G.add_node(sink_id, lat=destination[0], lon=destination[1])

    offsets = np.linspace(-lateral_spread_deg, lateral_spread_deg, n_lateral)
    columns: list[list[str]] = []

    for col_idx in range(n_longitudinal):
        gc_lat, gc_lon = gc_points[col_idx]
        brng = bearing(gc_lat, gc_lon, destination[0], destination[1])
        perp_bearing = (brng + 90) % 360

        col_nodes = []
        for lat_idx, offset in enumerate(offsets):
            offset_km = offset * 111.0
            nlat, nlon = destination_point(gc_lat, gc_lon, perp_bearing, offset_km)
            node_id = f"W{col_idx}_{lat_idx}"
            G.add_node(node_id, lat=nlat, lon=nlon)
            col_nodes.append(node_id)
        columns.append(col_nodes)

    for node_id in columns[0]:
        _add_edge(G, source_id, node_id, weather_field, traffic_field)

    for c in range(len(columns) - 1):
        for n1 in columns[c]:
            for n2 in columns[c + 1]:
                _add_edge(G, n1, n2, weather_field, traffic_field)

    for node_id in columns[-1]:
        _add_edge(G, node_id, sink_id, weather_field, traffic_field)

    return G


def _add_edge(G: nx.DiGraph, u: str, v: str, weather_field, traffic_field=None) -> None:
    """Add a directed edge with precomputed distance, bearing, weather, and traffic."""
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    dist = haversine(u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"])
    brng = bearing(u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"])
    mid_lat = (u_data["lat"] + v_data["lat"]) / 2
    mid_lon = (u_data["lon"] + v_data["lon"]) / 2
    wx = weather_field.sample(mid_lat, mid_lon)
    headwind = weather_field.headwind_along(mid_lat, mid_lon, brng)

    congestion = 0.0
    if traffic_field is not None:
        tx = traffic_field.sample(mid_lat, mid_lon)
        congestion = tx["congestion"]

    G.add_edge(u, v,
               distance_km=dist,
               bearing_deg=brng,
               mid_lat=mid_lat,
               mid_lon=mid_lon,
               turbulence_idx=wx["risk"],
               headwind_kts=headwind / 0.5144,
               temp_dev_c=wx["temp_dev_c"],
               congestion=congestion)
