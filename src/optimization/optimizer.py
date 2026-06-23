"""High-level route optimization orchestrator."""

from src.data.ingest import AIRCRAFT_SPECS
from src.optimization.astar import multi_objective_astar
from src.optimization.graph_builder import build_waypoint_graph
from src.optimization.rl_refiner import QLearningRefiner
from src.constants import CO2_KG_PER_KG_FUEL, CRUISE_SPEED_KMH, JET_FUEL_USD_PER_KG, KTS_TO_MS
from src.utils.geo import bearing, great_circle_path, haversine


class RouteOptimizer:
    """Generate and compare baseline, A*, and RL-refined route candidates."""

    def __init__(self, fuel_predictor, weather_field, config: dict, traffic_field=None):
        self.fuel_predictor = fuel_predictor
        self.weather_field = weather_field
        self.config = config
        self.traffic_field = traffic_field

    def optimize(
        self,
        origin_code: str,
        destination_code: str,
        aircraft: str,
        payload_kg: float,
        cruise_alt_ft: float,
        weights: dict,
        airports_dict: dict,
    ) -> list[dict]:
        """Return 3 route alternatives: baseline, A*, and A*+RL."""
        origin = tuple(airports_dict[origin_code])
        destination = tuple(airports_dict[destination_code])

        gc_cfg = self.config.get("optimizer", {}).get("graph", {})
        graph = build_waypoint_graph(
            origin, destination, self.weather_field,
            n_lateral=gc_cfg.get("n_lateral_offsets", 7),
            n_longitudinal=gc_cfg.get("n_longitudinal_steps", 25),
            lateral_spread_deg=gc_cfg.get("lateral_spread_deg", 6.0),
            traffic_field=self.traffic_field,
        )

        baseline = self._baseline_route(origin, destination, aircraft, payload_kg, cruise_alt_ft)

        astar_result = multi_objective_astar(
            graph, "SRC", "SNK", self.fuel_predictor,
            aircraft, payload_kg, cruise_alt_ft, weights, self.weather_field,
        )

        rl_cfg = self.config.get("optimizer", {}).get("rl_refiner", {})
        refiner = QLearningRefiner(
            n_episodes=rl_cfg.get("n_episodes", 200),
            epsilon=rl_cfg.get("epsilon", 0.2),
            alpha=rl_cfg.get("alpha", 0.1),
            gamma=rl_cfg.get("gamma", 0.95),
        )
        rl_result = refiner.refine(
            astar_result["path"], graph, self.fuel_predictor,
            self.weather_field, aircraft, payload_kg, cruise_alt_ft, weights,
        )

        jet_fuel_per_kg = JET_FUEL_USD_PER_KG
        co2_factor = CO2_KG_PER_KG_FUEL

        routes = [
            {
                "name": "Great Circle Baseline",
                "points": baseline["path"],
                "fuel_kg": baseline["total_fuel_kg"],
                "time_h": baseline["total_time_h"],
                "risk": baseline["mean_risk"],
                "congestion": self._mean_congestion(baseline["path"]),
                "cost_usd": baseline["total_fuel_kg"] * jet_fuel_per_kg,
                "co2_kg": baseline["total_fuel_kg"] * co2_factor,
                "color": "#888888",
            },
            {
                "name": "A* Fuel+Time Optimal",
                "points": astar_result["path"],
                "fuel_kg": astar_result["total_fuel_kg"],
                "time_h": astar_result["total_time_h"],
                "risk": astar_result["mean_risk"],
                "congestion": self._mean_congestion(astar_result["path"]),
                "cost_usd": astar_result["total_fuel_kg"] * jet_fuel_per_kg,
                "co2_kg": astar_result["total_fuel_kg"] * co2_factor,
                "color": "#4A90D9",
            },
            {
                "name": "A*+RL Risk-Aware",
                "points": rl_result["path"],
                "fuel_kg": rl_result["total_fuel_kg"],
                "time_h": rl_result["total_time_h"],
                "risk": rl_result["mean_risk"],
                "congestion": self._mean_congestion(rl_result["path"]),
                "cost_usd": rl_result["total_fuel_kg"] * jet_fuel_per_kg,
                "co2_kg": rl_result["total_fuel_kg"] * co2_factor,
                "color": "#2ECC71",
            },
        ]
        return routes

    def _baseline_route(
        self, origin: tuple, destination: tuple,
        aircraft: str, payload_kg: float, cruise_alt_ft: float,
    ) -> dict:
        """Straight great-circle route with segment-wise fuel estimation."""
        path = great_circle_path(origin[0], origin[1], destination[0], destination[1], n_points=30)
        total_fuel = 0.0
        total_time = 0.0
        risk_sum = 0.0

        for i in range(len(path) - 1):
            lat1, lon1 = path[i]
            lat2, lon2 = path[i + 1]
            dist = haversine(lat1, lon1, lat2, lon2)
            brng = bearing(lat1, lon1, lat2, lon2)
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            wx = self.weather_field.sample(mid_lat, mid_lon)
            headwind = self.weather_field.headwind_along(mid_lat, mid_lon, brng)

            fuel = self.fuel_predictor.predict_row(
                aircraft, dist, cruise_alt_ft, payload_kg,
                headwind / KTS_TO_MS, wx["temp_dev_c"], wx["risk"],
            )
            total_fuel += fuel
            total_time += dist / CRUISE_SPEED_KMH
            risk_sum += wx["risk"]

        n_seg = max(len(path) - 1, 1)
        return {
            "path": path,
            "total_fuel_kg": total_fuel,
            "total_time_h": total_time + 0.5,
            "mean_risk": risk_sum / n_seg,
        }

    def _mean_congestion(self, path: list[tuple[float, float]]) -> float:
        """Compute average traffic congestion along a route path."""
        if self.traffic_field is None or len(path) < 2:
            return 0.0
        total = 0.0
        for lat, lon in path:
            total += self.traffic_field.sample(lat, lon)["congestion"]
        return total / len(path)
