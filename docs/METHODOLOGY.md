# AeroOpti — Methodology

## Route Optimization: Why A* over Dijkstra

The waypoint graph can contain thousands of nodes (n_lateral × n_longitudinal). Dijkstra explores all reachable nodes uniformly, while A* uses a geographic heuristic (haversine distance to destination / cruise speed) to focus the search toward the target. This heuristic is admissible — it never overestimates the true remaining cost — guaranteeing optimality while typically expanding 40–60% fewer nodes than Dijkstra on transatlantic corridors.

## RL Refiner: Why Q-Learning over PPO

The lateral-offset action space is discrete and small (3 actions per waypoint), and the state space is naturally tabular (discretized lat/lon bins). Full policy-gradient methods like PPO add complexity (neural network policies, advantage estimation, clipping) without meaningful benefit in this setting. Tabular Q-learning converges faster, is fully interpretable (the Q-table can be inspected directly), and avoids hyperparameter sensitivity around network architecture.

## Fuel Model: Why XGBoost over Neural Networks

Fuel prediction is a tabular regression task with ~14 features. Tree-based gradient boosting consistently outperforms neural networks on tabular data of this size. XGBoost provides native feature importance and integrates with SHAP for post-hoc explainability, which is critical for dispatch decision support where operators need to understand *why* a fuel estimate differs between route segments.

## Delay Model: Why LSTM

Operational delays exhibit strong temporal autocorrelation — a delayed departure at an airport often cascades to subsequent flights on the same route. LSTMs are designed to capture sequential dependencies in time-series data. The sliding-window approach (10-flight sequences per origin-destination pair) lets the model learn how delays propagate and mean-revert over time.

## Cost Function

The multi-objective cost per edge combines four terms:

```
edge_cost = α × (fuel_kg / 10000) + β × time_h + γ × turbulence_idx + δ × (0.001 × distance_km)
```

- **α (fuel)**: Normalized by 10,000 kg to bring fuel into a comparable scale with time
- **β (time)**: Hours of flight time for the segment
- **γ (risk)**: Turbulence/convective risk index [0,1] — the primary safety lever
- **δ (airspace fee)**: Proxy for overflight charges, proportional to distance

Dispatchers adjust these weights through the dashboard to express operational priorities (cost-sensitive, schedule-critical, or safety-first).

## Weather Grid

The synthetic weather field uses Gaussian storm cells with spatial smoothing (σ=2 grid cells) to produce realistic spatial correlation. Real convective systems span hundreds of kilometers with smooth gradients at the edges — the Gaussian model captures this structure while remaining reproducible without external data dependencies.
