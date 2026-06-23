"""Tests for A* route optimizer."""

import pandas as pd

from src.data.ingest import generate_synthetic_flights
from src.data.weather import generate_weather_field
from src.models.fuel_predictor import FuelPredictor
from src.optimization.astar import multi_objective_astar
from src.optimization.graph_builder import build_waypoint_graph
from src.utils.geo import haversine


def _setup():
    """Train a quick fuel predictor and build a small graph."""
    df = generate_synthetic_flights(n=500, seed=42)
    fp = FuelPredictor(n_estimators=50, max_depth=4)
    fp.fit(df)
    wf = generate_weather_field(seed=42)

    origin = (40.6413, -73.7781)
    dest = (51.4700, -0.4543)
    graph = build_waypoint_graph(origin, dest, wf, n_lateral=5, n_longitudinal=10)
    return fp, wf, graph, origin, dest


def test_astar_finds_path():
    """A* should return a non-empty path."""
    fp, wf, graph, origin, dest = _setup()
    weights = {"fuel": 1.0, "time": 0.5, "risk": 2.0, "airspace_fee": 0.3}
    result = multi_objective_astar(graph, "SRC", "SNK", fp, "B777", 48375, 37000, weights, wf)
    assert len(result["path"]) > 2
    assert result["total_fuel_kg"] > 0


def test_astar_beats_baseline():
    """A* should produce a route with reasonable fuel relative to segment-wise baseline."""
    fp, wf, graph, origin, dest = _setup()
    weights = {"fuel": 1.0, "time": 0.5, "risk": 2.0, "airspace_fee": 0.3}
    result = multi_objective_astar(graph, "SRC", "SNK", fp, "B777", 48375, 37000, weights, wf)

    from src.utils.geo import great_circle_path, bearing as brng_fn
    path = great_circle_path(origin[0], origin[1], dest[0], dest[1], n_points=10)
    baseline_fuel = 0.0
    for i in range(len(path) - 1):
        d = haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        b = brng_fn(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        mid = ((path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2)
        hw = wf.headwind_along(mid[0], mid[1], b) / 0.5144
        wx = wf.sample(mid[0], mid[1])
        baseline_fuel += fp.predict_row("B777", d, 37000, 48375, hw, wx["temp_dev_c"], wx["risk"])
    assert result["total_fuel_kg"] <= baseline_fuel * 1.3


def test_cost_weights():
    """Higher risk weight should produce a route with different characteristics."""
    fp, wf, graph, _, _ = _setup()
    low_risk_w = {"fuel": 1.0, "time": 0.5, "risk": 0.1, "airspace_fee": 0.3}
    high_risk_w = {"fuel": 1.0, "time": 0.5, "risk": 5.0, "airspace_fee": 0.3}

    r_low = multi_objective_astar(graph, "SRC", "SNK", fp, "B777", 48375, 37000, low_risk_w, wf)
    r_high = multi_objective_astar(graph, "SRC", "SNK", fp, "B777", 48375, 37000, high_risk_w, wf)

    assert r_high["mean_risk"] <= r_low["mean_risk"] + 0.05
