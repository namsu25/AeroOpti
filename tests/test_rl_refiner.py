"""Tests for the Q-learning route refiner."""

from src.data.ingest import generate_synthetic_flights
from src.data.weather import generate_weather_field
from src.models.fuel_predictor import FuelPredictor
from src.optimization.astar import multi_objective_astar
from src.optimization.graph_builder import build_waypoint_graph
from src.optimization.rl_refiner import QLearningRefiner

_WEIGHTS = {"fuel": 1.0, "time": 0.5, "risk": 2.0, "airspace_fee": 0.3}


def _setup():
    df = generate_synthetic_flights(n=300, seed=11)
    fp = FuelPredictor(n_estimators=30, max_depth=3)
    fp.fit(df)
    wf = generate_weather_field(seed=11)
    origin = (40.6413, -73.7781)
    dest = (51.4700, -0.4543)
    graph = build_waypoint_graph(origin, dest, wf, n_lateral=5, n_longitudinal=10)
    astar_result = multi_objective_astar(graph, "SRC", "SNK", fp, "B777", 48375, 37000, _WEIGHTS, wf)
    return fp, wf, graph, astar_result


def test_refine_returns_valid_path():
    """Refined path should be non-empty with positive fuel and finite risk."""
    fp, wf, graph, astar_result = _setup()
    refiner = QLearningRefiner(n_episodes=20, epsilon=0.2, alpha=0.1, gamma=0.95)

    result = refiner.refine(
        astar_result["path"], graph, fp, wf, "B777", 48375, 37000, _WEIGHTS,
    )

    assert len(result["path"]) == len(astar_result["path"])
    assert result["total_fuel_kg"] > 0
    assert result["mean_risk"] >= 0


def test_refine_handles_short_path():
    """A path with fewer than 3 points should be returned as-is without training."""
    fp, wf, graph, _ = _setup()
    refiner = QLearningRefiner(n_episodes=20)
    short_path = [(40.6413, -73.7781), (51.4700, -0.4543)]

    result = refiner.refine(short_path, graph, fp, wf, "B777", 48375, 37000, _WEIGHTS)

    assert result["path"] == short_path
    assert result["total_fuel_kg"] > 0


def test_refine_is_deterministic_with_same_seed():
    """Same seed should produce the same refined path (reproducibility)."""
    fp, wf, graph, astar_result = _setup()
    r1 = QLearningRefiner(n_episodes=15, seed=99)
    r2 = QLearningRefiner(n_episodes=15, seed=99)

    result1 = r1.refine(astar_result["path"], graph, fp, wf, "B777", 48375, 37000, _WEIGHTS)
    result2 = r2.refine(astar_result["path"], graph, fp, wf, "B777", 48375, 37000, _WEIGHTS)

    assert result1["path"] == result2["path"]
