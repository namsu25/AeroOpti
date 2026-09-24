"""FastAPI backend serving the AeroOpti ML/optimization pipeline to the React frontend."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.cache import cached
from backend.schemas import OptimizeRequest, OptimizeResponse, Route
from src.data.ingest import AIRCRAFT_SPECS, AIRPORTS
from src.data.live_traffic import fetch_opensky_bbox, fetch_opensky_live
from src.data.live_weather import fetch_noaa_metars, fetch_noaa_sigmets, metars_to_dataframe
from src.data.traffic import generate_traffic_field
from src.data.weather import generate_weather_field
from src.models.fuel_predictor import FuelPredictor
from src.models.train import load_config
from src.optimization.optimizer import RouteNotFoundError, RouteOptimizer

app = FastAPI(title="AeroOpti API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = load_config()
_seed = _config["data"]["random_seed"]
_weather_field = generate_weather_field(seed=_seed)
_traffic_field = generate_traffic_field(seed=_seed)

def _records(df) -> list[dict]:
    """DataFrame -> JSON-safe records (OpenSky/METAR data can contain NaN, which
    plain `json.dumps` rejects and which otherwise surfaces as a 500 that drops
    CORS headers, misleadingly reported by browsers as a CORS failure)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


_model_path = PROJECT_ROOT / "models" / "fuel_predictor"
_fuel_predictor: FuelPredictor | None = (
    FuelPredictor.load(_model_path) if FuelPredictor.exists(_model_path) else None
)


@app.get("/api/config")
def get_config():
    """Static config: airports, aircraft specs, default cost weights."""
    return {
        "airports": AIRPORTS,
        "aircraft": AIRCRAFT_SPECS,
        "default_weights": _config.get("optimizer", {}).get("cost_weights", {}),
        "model_ready": _fuel_predictor is not None,
    }


@app.get("/api/weather-field")
def get_weather_field():
    """2D grids for the weather-risk heatmap and wind-vector layer."""
    wf = _weather_field
    return {
        "lat_grid": wf.lat_grid.tolist(),
        "lon_grid": wf.lon_grid.tolist(),
        "risk": wf.risk.tolist(),
        "wind_u": wf.wind_u.tolist(),
        "wind_v": wf.wind_v.tolist(),
        "temperature": wf.temperature.tolist(),
    }


@app.get("/api/traffic-field")
def get_traffic_field():
    """Traffic density grid plus the simulated major-corridor polylines."""
    tf = _traffic_field
    return {
        "lat_grid": tf.lat_grid.tolist(),
        "lon_grid": tf.lon_grid.tolist(),
        "density": tf.density.tolist(),
        "corridors": tf.corridors,
    }


@app.get("/api/live/traffic")
def get_live_traffic(
    min_lat: float | None = Query(None), min_lon: float | None = Query(None),
    max_lat: float | None = Query(None), max_lon: float | None = Query(None),
):
    """Live OpenSky aircraft positions, optionally scoped to a bounding box."""
    if None not in (min_lat, min_lon, max_lat, max_lon):
        key = f"traffic:{min_lat},{min_lon},{max_lat},{max_lon}"
        df = cached(key, 300, lambda: fetch_opensky_bbox((min_lat, min_lon), (max_lat, max_lon), margin_deg=0.0))
    else:
        df = cached("traffic:global", 300, fetch_opensky_live)
    return _records(df)


@app.get("/api/live/metars")
def get_live_metars():
    """Live NOAA METAR reports."""
    df = cached("metars", 600, lambda: metars_to_dataframe(fetch_noaa_metars()))
    return _records(df)


@app.get("/api/live/sigmets")
def get_live_sigmets():
    """Live NOAA SIGMET/AIRMET reports."""
    reports = cached("sigmets", 600, fetch_noaa_sigmets)
    return [r.__dict__ for r in reports]


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """Run the multi-objective route optimizer and return baseline/A*/A*+RL routes."""
    if _fuel_predictor is None:
        raise HTTPException(status_code=503, detail="Fuel model not trained. Run scripts/train_models.py.")
    if req.origin == req.destination:
        raise HTTPException(status_code=400, detail="Origin and destination must be different.")
    if req.origin not in AIRPORTS or req.destination not in AIRPORTS:
        raise HTTPException(status_code=400, detail="Unknown airport code.")
    if req.aircraft not in AIRCRAFT_SPECS:
        raise HTTPException(status_code=400, detail="Unknown aircraft type.")

    spec = AIRCRAFT_SPECS[req.aircraft]
    payload_kg = spec["payload_kg"] * req.payload_pct / 100
    cruise_alt_ft = spec["cruise_alt_ft"]

    optimizer = RouteOptimizer(_fuel_predictor, _weather_field, _config, traffic_field=_traffic_field)
    try:
        routes = optimizer.optimize(
            req.origin, req.destination, req.aircraft, payload_kg,
            cruise_alt_ft, req.weights.model_dump(), AIRPORTS,
        )
    except RouteNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    best_index = 1 + min(range(2), key=lambda i: routes[1 + i]["fuel_kg"])
    return OptimizeResponse(
        routes=[Route(**{**r, "points": [list(p) for p in r["points"]]}) for r in routes],
        best_index=best_index,
    )
