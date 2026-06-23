# AeroOpti

**AI-driven flight route optimizer and fuel consumption predictor for airline dispatch and operations.**

AeroOpti is a decision-support system that ingests aviation and weather data, predicts fuel burn and arrival delays using trained ML models, generates multi-objective route candidates, and surfaces human-centric trade-off decisions in an interactive Streamlit dashboard with MapLibre/OSM maps.

> **Note**: This is decision support — dispatchers and pilots retain final authority.

**Author**: Usman Mohammed

---

## Features

- **Fuel Prediction** — XGBoost model trained on synthetic BADA-style performance data (R2 ~ 0.998)
- **Delay Prediction** — PyTorch LSTM capturing temporal delay propagation patterns (MAE ~ 7 min)
- **Multi-Objective Route Optimization** — A* search over waypoint graphs with configurable cost weights
- **RL Risk Refinement** — Q-learning agent that reduces turbulence exposure over pure A*
- **MapLibre + OSM Maps** — Interactive pydeck maps on OpenStreetMap dark tiles
- **Live Data Sources** — OpenSky (traffic), Open-Meteo (winds), NOAA Aviation Weather (METARs/SIGMETs), OurAirports, OpenFlights
- **Fully Offline Mode** — Runs entirely on synthetic data with no API keys required

---

## Running in VS Code (Step by Step)

### Prerequisites

- **Python 3.10+** installed and on your PATH
- **VS Code** with the **Python extension** (ms-python.python) installed

### 1. Open the project

```
File > Open Folder > select the AeroOpti folder
```

Or from a terminal:

```bash
code "C:\Users\ZEPHYRUS G15\Desktop\AeroOpti"
```

### 2. Create a virtual environment

Open the VS Code integrated terminal (`Ctrl+`` ` or `Terminal > New Terminal`) and run:

```bash
python -m venv .venv
```

When VS Code detects the new environment, click **Yes** on the prompt to select it as your interpreter. If it doesn't prompt, press `Ctrl+Shift+P`, type **Python: Select Interpreter**, and pick the `.venv` one.

### 3. Activate the environment

The terminal should auto-activate. If not:

```powershell
# PowerShell (VS Code default on Windows)
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your terminal prompt.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, XGBoost, Streamlit, pydeck, and all other packages. Takes 2-5 minutes depending on your connection.

### 5. Generate sample data

```bash
python scripts/generate_sample_data.py
```

Expected output:
```
Generated 10000 flights, saved to data/sample/
```

### 6. Train the ML models

```bash
python scripts/train_models.py
```

Expected output:
```
Training fuel predictor (XGBoost)...
  R2=0.998  RMSE=1658  MAE=1103  MAPE=2.34%
Training delay predictor (LSTM)...
  MAE=6.81 min  RMSE=8.74 min
```

This creates `models/fuel_predictor.joblib` and `models/delay_predictor.pt`.

### 7. Test the optimizer (CLI)

```bash
python scripts/run_optimizer.py --origin JFK --destination LHR --aircraft B777
```

You should see a table comparing 3 routes (Baseline, A*, A*+RL) with fuel, time, risk, traffic, and cost columns.

### 8. Run the test suite

```bash
pytest tests/ -v
```

All 11 tests should pass.

### 9. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

This opens `http://localhost:8501` in your browser. The dashboard shows:

1. **Sidebar** — Select origin/destination, aircraft, payload %, cost weights, and scenario presets
2. **Route Map** — MapLibre dark map with route paths rendered on OpenStreetMap tiles
3. **Trade-off Charts** — Pareto scatter and grouped bar comparison
4. **Route Details Table** — Fuel, time, risk, congestion, CO2, cost per route
5. **Environment Layers** (6 tabs):
   - Weather Risk (heatmap)
   - Wind Field (arc vectors)
   - Traffic Corridors (arc layer)
   - Live Traffic (OpenSky aircraft, requires "Enable Live Data" toggle)
   - NOAA METARs (station weather, requires "Enable Live Data" toggle)
   - Weather Dashboard (4-panel Plotly overview)

Toggle **Enable Live Data** in the sidebar to fetch real-time data from OpenSky and NOAA. No API keys needed.

To stop the server, press `Ctrl+C` in the terminal.

---

## Project Structure

```
AeroOpti/
├── configs/default.yaml          # Hyperparameters, airport coords, cost weights
├── data/sample/                  # Generated synthetic flights (parquet)
├── src/
│   ├── data/
│   │   ├── ingest.py             # Synthetic flight generator + OpenSky loader
│   │   ├── weather.py            # Simulated weather field
│   │   ├── traffic.py            # Simulated traffic corridors
│   │   ├── live_traffic.py       # OpenSky Network live API
│   │   ├── live_weather.py       # Open-Meteo + NOAA Aviation Weather APIs
│   │   ├── airports_db.py        # OurAirports + OpenFlights data
│   │   └── preprocess.py         # Feature engineering for ML models
│   ├── models/
│   │   ├── fuel_predictor.py     # XGBoost wrapper
│   │   ├── delay_predictor.py    # PyTorch LSTM wrapper
│   │   └── train.py              # Unified training entrypoint
│   ├── optimization/
│   │   ├── graph_builder.py      # Waypoint graph over great-circle corridor
│   │   ├── astar.py              # Multi-objective A* search
│   │   ├── rl_refiner.py         # Q-learning risk refiner
│   │   └── optimizer.py          # High-level orchestration
│   └── utils/
│       ├── geo.py                # Haversine, bearing, great-circle math
│       ├── metrics.py            # RMSE, MAE, MAPE, R2
│       ├── maplibre.py           # pydeck + MapLibre + OSM tile maps
│       └── viz.py                # Plotly charts (Pareto, bars, weather panels)
├── app/streamlit_app.py          # Interactive dashboard
├── scripts/                      # CLI entrypoints
├── tests/                        # pytest suite (11 tests)
├── models/                       # Saved model artifacts (after training)
└── docs/                         # Methodology and evaluation
```

---

## Data Sources

| Source | What it provides | API key needed |
|--------|-----------------|----------------|
| **OpenSky Network** | Live aircraft positions (ICAO24, lat, lon, altitude, speed, track) | No (anonymous, rate-limited) |
| **Open-Meteo** | Upper-level winds and temperature at FL340 (250hPa) | No |
| **NOAA Aviation Weather** | METARs, TAFs, SIGMETs, AIRMETs, PIREPs | No |
| **OurAirports** | 3,000+ airports with IATA codes, coordinates, elevation | No (CSV download) |
| **OpenFlights** | 67,000+ airline routes between airports | No (CSV download) |
| **Synthetic Generator** | 10,000 flights with realistic fuel/delay correlations | N/A (built-in) |

All external APIs are optional. The project runs fully offline using the synthetic data generator.

---

## Troubleshooting

**"Fuel prediction model not found"** — Run steps 5 and 6 first (generate data, then train models).

**PowerShell execution policy error** — Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and try activating the venv again.

**`pip install` fails on torch** — Make sure you have Python 3.10+ (not 3.13). Check with `python --version`.

**OpenSky returns empty** — The anonymous API is rate-limited (~100 requests/day). Wait a few minutes or proceed without live data.

**Port 8501 already in use** — Run `streamlit run app/streamlit_app.py --server.port 8502` to use a different port.

---

## Configuration

All parameters live in `configs/default.yaml`:

- Model hyperparameters (XGBoost trees, LSTM hidden size, etc.)
- Optimization graph resolution and cost weights (fuel, time, risk, traffic, airspace fee)
- Airport coordinates
- RL refiner episodes, epsilon, alpha, gamma

---

## License

MIT — see [LICENSE](LICENSE).
