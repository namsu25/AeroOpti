"""End-to-end tests for the RouteOptimizer orchestrator."""

from src.data.ingest import AIRPORTS, generate_synthetic_flights
from src.data.traffic import generate_traffic_field
from src.data.weather import generate_weather_field
from src.models.fuel_predictor import FuelPredictor
from src.optimization.optimizer import RouteNotFoundError, RouteOptimizer

_CONFIG = {
    "optimizer": {
        "graph": {"n_lateral_offsets": 5, "n_longitudinal_steps": 8, "lateral_spread_deg": 6.0},
        "rl_refiner": {"n_episodes": 20, "epsilon": 0.2, "alpha": 0.1, "gamma": 0.95},
    }
}
_WEIGHTS = {"fuel": 1.0, "time": 0.5, "risk": 2.0, "airspace_fee": 0.3, "traffic": 0.5}


def _fuel_predictor() -> FuelPredictor:
    df = generate_synthetic_flights(n=300, seed=7)
    fp = FuelPredictor(n_estimators=30, max_depth=3)
    fp.fit(df)
    return fp


def test_optimize_returns_three_valid_routes():
    """optimize() should return baseline, A*, and A*+RL routes with positive fuel/time."""
    fp = _fuel_predictor()
    wf = generate_weather_field(seed=7)
    tf = generate_traffic_field(seed=7)
    optimizer = RouteOptimizer(fp, wf, _CONFIG, traffic_field=tf)

    routes = optimizer.optimize(
        "JFK", "LHR", "B777", 48375, 37000, _WEIGHTS, AIRPORTS,
    )

    assert len(routes) == 3
    names = {r["name"] for r in routes}
    assert names == {"Great Circle Baseline", "A* Fuel+Time Optimal", "A*+RL Risk-Aware"}
    for r in routes:
        assert r["fuel_kg"] > 0
        assert r["time_h"] > 0
        assert len(r["points"]) >= 2
        assert 0.0 <= r["congestion"] <= 1.0


def test_optimize_handles_zero_length_route_without_crashing():
    """Origin == destination degenerates to a zero-length graph.

    The optimizer should either raise the documented RouteNotFoundError or
    return finite, non-negative metrics -- never an unhandled exception like
    a bare ZeroDivisionError leaking out of internal cost math.
    """
    fp = _fuel_predictor()
    wf = generate_weather_field(seed=7)
    optimizer = RouteOptimizer(fp, wf, _CONFIG)

    try:
        routes = optimizer.optimize("JFK", "JFK", "B777", 48375, 37000, _WEIGHTS, AIRPORTS)
    except RouteNotFoundError:
        return

    for r in routes:
        assert r["fuel_kg"] == r["fuel_kg"]  # not NaN
        assert r["fuel_kg"] >= 0
