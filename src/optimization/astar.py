"""Multi-objective A* search over the waypoint graph."""

import heapq

import networkx as nx

from src.constants import CRUISE_SPEED_KMH
from src.utils.geo import haversine


def multi_objective_astar(
    graph: nx.DiGraph,
    source: str,
    target: str,
    fuel_predictor,
    aircraft: str,
    payload_kg: float,
    cruise_alt_ft: float,
    weights: dict,
    weather_field,
) -> dict:
    """Run A* with a composite cost function and return the optimal route."""
    open_set: list[tuple[float, int, str]] = []
    counter = 0
    g_score = {source: 0.0}
    came_from: dict[str, str | None] = {source: None}
    fuel_accum: dict[str, float] = {source: 0.0}
    time_accum: dict[str, float] = {source: 0.0}
    risk_accum: dict[str, float] = {source: 0.0}
    risk_count: dict[str, int] = {source: 0}

    target_data = graph.nodes[target]
    h_val = _heuristic(graph.nodes[source], target_data, weights)
    heapq.heappush(open_set, (h_val, counter, source))

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == target:
            path = _reconstruct(came_from, current, graph)
            rc = risk_count[current] if risk_count[current] > 0 else 1
            return {
                "path": path,
                "total_fuel_kg": fuel_accum[current],
                "total_time_h": time_accum[current],
                "mean_risk": risk_accum[current] / rc,
                "total_cost": g_score[current],
                "n_waypoints": len(path),
            }

        for neighbor in graph.successors(current):
            edge = graph.edges[current, neighbor]
            fuel = fuel_predictor.predict_row(
                aircraft, edge["distance_km"], cruise_alt_ft,
                payload_kg, edge["headwind_kts"],
                edge["temp_dev_c"], edge["turbulence_idx"],
            )
            time_h = edge["distance_km"] / CRUISE_SPEED_KMH
            risk_val = edge["turbulence_idx"]
            fee = 0.001 * edge["distance_km"]
            congestion = edge.get("congestion", 0.0)

            edge_cost = (
                weights.get("fuel", 1.0) * fuel / 10000
                + weights.get("time", 0.5) * time_h
                + weights.get("risk", 2.0) * risk_val
                + weights.get("airspace_fee", 0.3) * fee
                + weights.get("traffic", 0.0) * congestion
            )

            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                fuel_accum[neighbor] = fuel_accum[current] + fuel
                time_accum[neighbor] = time_accum[current] + time_h
                risk_accum[neighbor] = risk_accum[current] + risk_val
                risk_count[neighbor] = risk_count[current] + 1

                h = _heuristic(graph.nodes[neighbor], target_data, weights)
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, neighbor))

    return {
        "path": [], "total_fuel_kg": 0, "total_time_h": 0,
        "mean_risk": 0, "total_cost": float("inf"), "n_waypoints": 0,
    }


def _heuristic(node_data: dict, target_data: dict, weights: dict) -> float:
    """Admissible heuristic: time-based lower bound to destination."""
    d = haversine(node_data["lat"], node_data["lon"], target_data["lat"], target_data["lon"])
    return weights.get("time", 0.5) * d / CRUISE_SPEED_KMH


def _reconstruct(came_from: dict, current: str, graph: nx.DiGraph) -> list[tuple[float, float]]:
    """Trace back the path from target to source."""
    path = []
    while current is not None:
        data = graph.nodes[current]
        path.append((data["lat"], data["lon"]))
        current = came_from.get(current)
    path.reverse()
    return path
