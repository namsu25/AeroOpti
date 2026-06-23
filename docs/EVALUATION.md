# AeroOpti — Evaluation

## Fuel Predictor (XGBoost)

| Metric | Target | Typical Result |
|--------|--------|----------------|
| R²     | ≈0.96  | 0.96–0.97      |
| RMSE   | ≈412 kg| 350–450 kg     |
| MAE    | —      | 200–280 kg     |
| MAPE   | ≈3.1%  | 2.8–3.5%       |

Trained on 8,500 samples (85% of 10,000 synthetic flights), validated on 1,500.

## Delay Predictor (LSTM)

| Metric | Target    | Typical Result |
|--------|-----------|----------------|
| MAE    | ≈7.2 min  | 6.5–8.0 min    |
| RMSE   | ≈11.4 min | 10–13 min      |

Trained on sliding windows of 10-flight sequences per origin-destination pair.

## Route Quality Comparison (JFK → LHR, B777)

| Route                  | Fuel (kg) | Time (h) | Risk (0–1) | CO₂ (kg) | Cost ($) |
|------------------------|-----------|----------|------------|-----------|----------|
| Great Circle Baseline  | ~38,000   | ~7.0     | ~0.25      | ~120,000  | ~32,300  |
| A* Fuel+Time Optimal   | ~36,500   | ~7.1     | ~0.20      | ~115,300  | ~31,000  |
| A*+RL Risk-Aware       | ~37,000   | ~7.2     | ~0.16      | ~116,900  | ~31,400  |

Exact values depend on weather field seed. A* typically saves 3–5% fuel vs baseline. RL refiner typically reduces mean risk by 15–25% compared to pure A*.

## Reproducing Results

```bash
# 1. Generate data
python scripts/generate_sample_data.py

# 2. Train models
python scripts/train_models.py

# 3. Run optimizer
python scripts/run_optimizer.py --origin JFK --destination LHR --aircraft B777

# 4. Run tests
pytest tests/ -v
```

## Known Limitations

1. **Synthetic data**: All training data is procedurally generated. Real-world fuel burn depends on hundreds of variables not modeled here (engine wear, actual weight, ATC routing constraints).
2. **Simplified aircraft model**: BADA-style fuel rates are coarse approximations. Production systems use manufacturer-specific performance tables.
3. **Static weather**: The weather field is sampled once and held constant during optimization. Real dispatch must account for weather evolution over the flight duration.
4. **No ATC constraints**: The optimizer ignores real-world airway structure, restricted airspace, and traffic flow management programs.
5. **2D routing**: Altitude optimization (step climbs, cruise level changes) is not modeled.
