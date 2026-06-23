"""Q-learning refiner that reduces risk over stochastic weather rollouts."""

from collections import defaultdict

import numpy as np

from src.constants import CRUISE_SPEED_KMH, KTS_TO_MS
from src.utils.geo import bearing, destination_point, haversine


_ACTIONS = [-2, -1, 0, 1, 2]
_N_ACTIONS = len(_ACTIONS)
_LATERAL_STEP_KM = 60.0


class QLearningRefiner:
    """Tabular Q-learning that refines an A* path by adjusting lateral offsets."""

    def __init__(
        self,
        n_episodes: int = 200,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.95,
        seed: int = 42,
    ):
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(_N_ACTIONS))

    def refine(
        self,
        base_path: list[tuple[float, float]],
        graph,
        fuel_predictor,
        weather_field,
        aircraft: str,
        payload_kg: float,
        cruise_alt_ft: float,
        weights: dict,
    ) -> dict:
        """Run Q-learning episodes to find a lower-risk path variant."""
        if len(base_path) < 3:
            return _path_to_result(base_path, fuel_predictor, weather_field,
                                   aircraft, payload_kg, cruise_alt_ft)

        # --- Training phase: explore stochastic weather rollouts ---
        for episode in range(self.n_episodes):
            perturb = self.rng.uniform(0.8, 1.2, size=weather_field.risk.shape)
            current_path = list(base_path)

            for step_idx in range(1, len(current_path) - 1):
                state = (step_idx, _discretize(current_path[step_idx]))

                if self.rng.random() < self.epsilon:
                    action_idx = self.rng.integers(_N_ACTIONS)
                else:
                    action_idx = int(np.argmax(self.q_table[state]))

                current_path[step_idx] = _apply_action(
                    current_path[step_idx], base_path[-1],
                    _ACTIONS[action_idx],
                )

                wx = weather_field.sample(*current_path[step_idx])
                risk = wx["risk"] * float(perturb[
                    min(int((current_path[step_idx][0] + 80) / 2), perturb.shape[0] - 1),
                    min(int((current_path[step_idx][1] + 180) / 2), perturb.shape[1] - 1),
                ])
                reward = -weights.get("risk", 2.0) * risk * 5.0

                next_state = (step_idx + 1, _discretize(
                    current_path[min(step_idx + 1, len(current_path) - 1)]
                ))
                best_next = float(np.max(self.q_table[next_state]))
                old = self.q_table[state][action_idx]
                self.q_table[state][action_idx] = (
                    old + self.alpha * (reward + self.gamma * best_next - old)
                )

        # --- Exploitation phase: build path from trained Q-table ---
        refined_path = list(base_path)
        for step_idx in range(1, len(refined_path) - 1):
            state = (step_idx, _discretize(refined_path[step_idx]))
            action_idx = int(np.argmax(self.q_table[state]))
            refined_path[step_idx] = _apply_action(
                refined_path[step_idx], base_path[-1],
                _ACTIONS[action_idx],
            )

        return _path_to_result(refined_path, fuel_predictor, weather_field,
                               aircraft, payload_kg, cruise_alt_ft)


def _apply_action(
    point: tuple[float, float],
    dest: tuple[float, float],
    offset: int,
) -> tuple[float, float]:
    """Shift a waypoint laterally by offset steps. Returns the new position."""
    if offset == 0:
        return point
    lat, lon = point
    brng = bearing(lat, lon, dest[0], dest[1])
    perp = (brng + 90 * (1 if offset > 0 else -1)) % 360
    return destination_point(lat, lon, perp, _LATERAL_STEP_KM * abs(offset))


def _discretize(point: tuple[float, float]) -> tuple[int, int]:
    """Discretize lat/lon to 1-degree bins for Q-table keys."""
    return (round(point[0]), round(point[1]))


def _path_to_result(
    path: list[tuple[float, float]],
    fuel_predictor,
    weather_field,
    aircraft: str,
    payload_kg: float,
    cruise_alt_ft: float,
) -> dict:
    """Evaluate a path end-to-end: compute fuel, time, and risk."""
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
        wx = weather_field.sample(mid_lat, mid_lon)
        headwind = weather_field.headwind_along(mid_lat, mid_lon, brng)

        fuel = fuel_predictor.predict_row(
            aircraft, dist, cruise_alt_ft, payload_kg,
            headwind / KTS_TO_MS, wx["temp_dev_c"], wx["risk"],
        )
        total_fuel += fuel
        total_time += dist / CRUISE_SPEED_KMH
        risk_sum += wx["risk"]

    n_segments = max(len(path) - 1, 1)
    return {
        "path": path,
        "total_fuel_kg": total_fuel,
        "total_time_h": total_time + 0.5,
        "mean_risk": risk_sum / n_segments,
        "total_cost": 0.0,
        "n_waypoints": len(path),
    }
