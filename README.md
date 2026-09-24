# AeroOpti

**AI-driven flight route optimizer and fuel consumption predictor for airline dispatch and operations.**

AeroOpti is a decision-support system that ingests aviation and weather data, predicts fuel burn and arrival delays using trained ML models, generates multi-objective route candidates, and surfaces human-centric trade-off decisions in an interactive React dashboard with a real MapLibre GL JS map.

The app is now split into a **FastAPI backend** (`backend/`, wraps the existing Python ML/optimization pipeline as a JSON API) and a **React + TypeScript frontend** (`frontend/`, Vite + MapLibre GL JS + Recharts). The original Streamlit app (`app/streamlit_app.py`) still works and is kept as a lightweight alternative, but the React app is the primary UI going forward.

> **Note**: This is decision support — dispatchers and pilots retain final authority.

**Author**: Usman Mohammed

---

## Features

- **Fuel Prediction** — XGBoost model trained on synthetic BADA-style performance data (see the R² caveat in [docs/EVALUATION.md](docs/EVALUATION.md))
- **Delay Prediction** — PyTorch LSTM capturing temporal delay propagation patterns (MAE ~ 7 min)
- **Multi-Objective Route Optimization** — A* search over waypoint graphs with configurable cost weights
- **RL Risk Refinement** — Q-learning agent that reduces turbulence exposure over pure A*
- **Real Interactive Map** — MapLibre GL JS (vector basemap, pan/zoom/hover tooltips) with toggleable layers: routes, weather-risk heatmap, wind field, traffic corridors, live traffic, NOAA METARs — all on one map instead of separate embeds
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

This creates `models/fuel_predictor.xgb.json` + `models/fuel_predictor.meta.json` (native XGBoost format, no pickle) and `models/delay_predictor.pt`, plus `models/latest_metrics.json` and `models/training_history.jsonl` recording each run's metrics, config, and git commit.

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

### 9. Launch the backend API

```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

This serves the JSON API at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`). It loads the trained fuel model and generates the weather/traffic fields once at startup.

### 10. Launch the React dashboard

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

This opens `http://localhost:5173` in your browser (it expects the backend on `http://localhost:8000`; override with a `VITE_API_BASE` env var if needed). The dashboard shows:

1. **Sidebar** — Select origin/destination, aircraft, payload %, cost weights, and scenario presets
2. **KPI cards** — Cost/fuel/CO2 savings and time delta of the best route vs. the great-circle baseline
3. **Route Map** — One interactive MapLibre GL map with toggleable layers: routes (solid = optimized, dashed = baseline), airports, weather-risk heatmap, wind field, simulated traffic corridors, live OpenSky traffic, and NOAA METARs (hover any route/aircraft/station for details)
4. **Trade-off Charts** — Pareto scatter and grouped bar comparison (Recharts)
5. **Route Details Table** — Fuel, time, risk, congestion, CO2, cost per route

Toggle **Enable Live Data** in the sidebar to fetch real-time traffic/METARs from OpenSky and NOAA (polled every 60s). No API keys needed.

To stop either server, press `Ctrl+C` in its terminal.

### Optional: the legacy Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Still works standalone (no backend/frontend needed) and opens `http://localhost:8501`. Kept as a lightweight fallback; the React app is where new UI work happens.

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
│       ├── maplibre.py           # pydeck + MapLibre + OSM tile maps (legacy Streamlit app)
│       └── viz.py                # Plotly charts (legacy Streamlit app)
├── backend/                      # FastAPI wrapper around src/ (JSON API for the React app)
│   ├── main.py                   # Routes: /api/config, /api/optimize, /api/weather-field, /api/live/*
│   ├── schemas.py                # Pydantic request/response models
│   └── cache.py                  # TTL cache for live-data endpoints
├── frontend/                     # React + TypeScript + Vite dashboard (primary UI)
│   └── src/
│       ├── App.tsx               # Page layout and state
│       ├── api.ts                # Backend client
│       └── components/
│           ├── RouteMap.tsx      # MapLibre GL map with all toggleable layers
│           ├── Sidebar.tsx, KpiCards.tsx, Charts.tsx, RouteTable.tsx, LayerToggles.tsx
├── app/streamlit_app.py          # Legacy standalone dashboard (still functional)
├── scripts/                      # CLI entrypoints
├── tests/                        # pytest suite (28 tests)
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

**Frontend shows a CORS error in the browser console** — This almost always means the backend returned a 500 (an unhandled server error drops CORS headers, which browsers then misreport as a CORS failure) rather than an actual CORS misconfiguration. Check the `uvicorn` terminal for a traceback.

**Frontend can't reach the API / requests fail** — Make sure `python -m uvicorn backend.main:app --port 8000` is running before `npm run dev`; the frontend defaults to `http://localhost:8000` (override with `VITE_API_BASE`).

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
