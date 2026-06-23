"""AeroOpti interactive Streamlit dashboard with MapLibre maps and live data."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import yaml

from src.data.ingest import AIRCRAFT_SPECS
from src.data.weather import generate_weather_field
from src.data.traffic import generate_traffic_field
from src.data.live_traffic import fetch_opensky_live, fetch_opensky_bbox
from src.data.live_weather import (
    fetch_noaa_metars, fetch_noaa_sigmets, metars_to_dataframe,
)
from src.models.fuel_predictor import FuelPredictor
from src.optimization.optimizer import RouteOptimizer
from src.utils.viz import plot_pareto, plot_comparison_bars, plot_weather_dashboard
from src.utils.maplibre import (
    route_map, live_traffic_map, weather_risk_map,
    wind_barb_map, metar_station_map, traffic_corridors_map,
)

st.set_page_config(page_title="AeroOpti", layout="wide")


@st.cache_resource
def load_fuel_model():
    """Load the trained fuel predictor."""
    path = PROJECT_ROOT / "models" / "fuel_predictor.joblib"
    if not path.exists():
        return None
    return FuelPredictor.load(path)


@st.cache_data(ttl=3600)
def get_weather_field(seed: int = 42):
    """Generate and cache the weather field."""
    return generate_weather_field(seed=seed)


@st.cache_data(ttl=3600)
def get_traffic_field(seed: int = 42):
    """Generate and cache the traffic field."""
    return generate_traffic_field(seed=seed)


@st.cache_data(ttl=300)
def get_live_traffic():
    """Fetch live traffic from OpenSky (cached 5 min)."""
    return fetch_opensky_live()


@st.cache_data(ttl=600)
def get_live_metars():
    """Fetch METAR reports from NOAA (cached 10 min)."""
    return metars_to_dataframe(fetch_noaa_metars())


@st.cache_data(ttl=600)
def get_live_sigmets():
    """Fetch active SIGMETs from NOAA (cached 10 min)."""
    return fetch_noaa_sigmets()


def load_config():
    """Load default config."""
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Render the full AeroOpti dashboard."""
    config = load_config()
    airports = config["airports"]
    airport_codes = sorted(airports.keys())

    # --- Sidebar ---
    st.sidebar.title("AeroOpti")
    st.sidebar.header("Route Configuration")

    origin = st.sidebar.selectbox("Origin", airport_codes, index=airport_codes.index("JFK"))
    destination = st.sidebar.selectbox("Destination", airport_codes, index=airport_codes.index("LHR"))
    aircraft = st.sidebar.selectbox("Aircraft Type", sorted(AIRCRAFT_SPECS.keys()), index=6)
    payload_pct = st.sidebar.slider("Payload %", 50, 100, 75)

    spec = AIRCRAFT_SPECS[aircraft]
    payload_kg = spec["payload_kg"] * payload_pct / 100
    cruise_alt_ft = spec["cruise_alt_ft"]

    st.sidebar.subheader("Scenario Presets")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("Fuel-First"):
        st.session_state["w_fuel"] = 3.0
        st.session_state["w_time"] = 0.3
        st.session_state["w_risk"] = 1.0
        st.session_state["w_fee"] = 0.3
        st.session_state["w_traffic"] = 0.2
    if col2.button("Time-Sensitive"):
        st.session_state["w_fuel"] = 0.5
        st.session_state["w_time"] = 3.0
        st.session_state["w_risk"] = 0.5
        st.session_state["w_fee"] = 0.3
        st.session_state["w_traffic"] = 0.5
    if col3.button("Low-Risk"):
        st.session_state["w_fuel"] = 1.0
        st.session_state["w_time"] = 0.5
        st.session_state["w_risk"] = 5.0
        st.session_state["w_fee"] = 0.3
        st.session_state["w_traffic"] = 1.0

    with st.sidebar.expander("Advanced -- Cost Weights"):
        w_fuel = st.slider("Fuel weight", 0.1, 3.0, 1.0, 0.1, key="w_fuel")
        w_time = st.slider("Time weight", 0.1, 3.0, 0.5, 0.1, key="w_time")
        w_risk = st.slider("Risk weight", 0.1, 5.0, 2.0, 0.1, key="w_risk")
        w_fee = st.slider("Airspace fee", 0.0, 1.0, 0.3, 0.1, key="w_fee")
        w_traffic = st.slider("Traffic avoidance", 0.0, 3.0, 0.5, 0.1, key="w_traffic")

    st.sidebar.markdown("---")
    live_data = st.sidebar.toggle("Enable Live Data", value=False,
                                   help="Fetch real-time traffic from OpenSky and weather from NOAA")

    weights = {"fuel": w_fuel, "time": w_time, "risk": w_risk, "airspace_fee": w_fee, "traffic": w_traffic}
    run_clicked = st.sidebar.button("Optimize Routes", type="primary", use_container_width=True)

    # --- Main Area ---
    st.title("AeroOpti -- AI Flight Route Optimizer")

    fp = load_fuel_model()
    if fp is None:
        st.warning(
            "Fuel prediction model not found. Run:\n\n"
            "```bash\npython scripts/generate_sample_data.py\npython scripts/train_models.py\n```"
        )
        return

    if origin == destination:
        st.error("Origin and destination must be different.")
        return

    wf = get_weather_field(config["data"]["random_seed"])
    tf = get_traffic_field(config["data"]["random_seed"])

    # --- Pre-optimization: show live data panels ---
    if not run_clicked:
        st.info("Configure your route in the sidebar and click **Optimize Routes** to begin.")
        st.markdown("---")

        if live_data:
            _render_live_panels(airports, wf, tf)
        else:
            st.subheader("Global Traffic Corridors")
            st.pydeck_chart(traffic_corridors_map(tf, airports=airports))
            st.caption("Simulated traffic corridors. Toggle **Enable Live Data** in the sidebar for real-time OpenSky traffic.")
        return

    # --- Run optimization ---
    optimizer = RouteOptimizer(fp, wf, config, traffic_field=tf)

    with st.spinner("Running multi-objective optimization..."):
        routes = optimizer.optimize(
            origin, destination, aircraft, payload_kg,
            cruise_alt_ft, weights, airports,
        )

    baseline = routes[0]
    best = min(routes[1:], key=lambda r: r["fuel_kg"])

    # Row 1 -- KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost Saving",
              f"${baseline['cost_usd'] - best['cost_usd']:,.0f}",
              f"{(baseline['cost_usd'] - best['cost_usd']) / baseline['cost_usd'] * 100:.1f}%")
    c2.metric("Fuel Saving",
              f"{baseline['fuel_kg'] - best['fuel_kg']:,.0f} kg",
              f"{(baseline['fuel_kg'] - best['fuel_kg']) / baseline['fuel_kg'] * 100:.1f}%")
    c3.metric("CO2 Avoided",
              f"{baseline['co2_kg'] - best['co2_kg']:,.0f} kg",
              f"{(baseline['co2_kg'] - best['co2_kg']) / baseline['co2_kg'] * 100:.1f}%")
    time_delta_min = (best['time_h'] - baseline['time_h']) * 60
    c4.metric("Time Delta", f"{time_delta_min:+.0f} min")

    # Row 2 -- Route map (MapLibre)
    st.subheader("Route Map")
    st.pydeck_chart(route_map(routes, origin, destination, airports))

    # Row 3 -- Pareto + bars (keep Plotly for non-geo charts)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_pareto(routes), use_container_width=True)
    with right:
        st.plotly_chart(plot_comparison_bars(routes), use_container_width=True)

    # Row 4 -- Route details table
    st.subheader("Route Details")
    table_data = []
    for r in routes:
        table_data.append({
            "Route": r["name"],
            "Fuel (kg)": f"{r['fuel_kg']:,.0f}",
            "Time (h)": f"{r['time_h']:.2f}",
            "Risk (0-1)": f"{r['risk']:.3f}",
            "Congestion (0-1)": f"{r['congestion']:.3f}",
            "CO2 (kg)": f"{r['co2_kg']:,.0f}",
            "Est. Cost ($)": f"{r['cost_usd']:,.0f}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Row 5 -- Environment layers (MapLibre tabs)
    all_lats = [p[0] for r in routes for p in r["points"]]
    all_lons = [p[1] for r in routes for p in r["points"]]
    lat_bounds = (min(all_lats) - 10, max(all_lats) + 10)
    lon_bounds = (min(all_lons) - 10, max(all_lons) + 10)

    st.subheader("Environment Layers")
    tabs = st.tabs(["Weather Risk", "Wind Field", "Traffic Corridors", "Live Traffic", "NOAA METARs", "Weather Dashboard"])

    with tabs[0]:
        st.pydeck_chart(weather_risk_map(wf, routes, lat_bounds, lon_bounds))
        st.caption("Heatmap of turbulence/convective risk. Brighter = higher risk.")

    with tabs[1]:
        st.pydeck_chart(wind_barb_map(wf, routes, lat_bounds, lon_bounds))
        st.caption("Arcs show wind direction and speed. Red/blue = faster/slower winds.")

    with tabs[2]:
        st.pydeck_chart(traffic_corridors_map(tf, routes, airports))
        st.caption("Orange arcs = simulated major traffic corridors (width = intensity).")

    with tabs[3]:
        if live_data:
            with st.spinner("Fetching live traffic from OpenSky Network..."):
                o_coords = tuple(airports[origin])
                d_coords = tuple(airports[destination])
                live_df = fetch_opensky_bbox(o_coords, d_coords, margin_deg=8.0)
            if live_df.empty:
                st.info("No live traffic data available (OpenSky may be rate-limited). Try again in a few minutes.")
            else:
                st.pydeck_chart(live_traffic_map(live_df, routes, airports))
                st.caption(f"Showing {len(live_df)} live aircraft from OpenSky Network in the route corridor.")
        else:
            st.info("Toggle **Enable Live Data** in the sidebar to fetch real-time aircraft positions from OpenSky Network.")

    with tabs[4]:
        if live_data:
            with st.spinner("Fetching METARs from NOAA Aviation Weather..."):
                metars_df = get_live_metars()
            if metars_df.empty:
                st.info("No METAR data available from NOAA.")
            else:
                st.pydeck_chart(metar_station_map(metars_df, routes))
                cat_legend = "Green = VFR | Blue = MVFR | Red = IFR | Purple = LIFR. Bubble size = wind speed."
                st.caption(f"METAR stations colored by flight category. {cat_legend}")
                with st.expander("Raw METAR reports"):
                    st.dataframe(metars_df[["station", "category", "wind_kts", "visibility_mi", "ceiling_ft", "temp_c"]].head(50),
                                 use_container_width=True, hide_index=True)

                sigmets = get_live_sigmets()
                if sigmets:
                    with st.expander(f"Active SIGMETs/AIRMETs ({len(sigmets)})"):
                        for sig in sigmets[:20]:
                            st.text(sig.raw_text[:200])
        else:
            st.info("Toggle **Enable Live Data** in the sidebar to fetch NOAA aviation weather (METARs, SIGMETs).")

    with tabs[5]:
        st.plotly_chart(
            plot_weather_dashboard(wf, lat_bounds, lon_bounds, routes),
            use_container_width=True,
        )

    st.caption(
        "AeroOpti -- Decision support only. Pilots and dispatchers retain "
        "final authority. Not for operational use."
    )


def _render_live_panels(airports: dict, wf, tf):
    """Render pre-optimization live data panels."""
    tab_traffic, tab_metars, tab_corridors = st.tabs(["Live Traffic", "NOAA METARs", "Traffic Corridors"])

    with tab_traffic:
        with st.spinner("Fetching live aircraft from OpenSky Network..."):
            live_df = get_live_traffic()
        if live_df.empty:
            st.info("No live traffic data available (OpenSky may be rate-limited or offline).")
        else:
            st.pydeck_chart(live_traffic_map(live_df, airports=airports))
            st.caption(f"{len(live_df)} aircraft currently tracked by OpenSky Network.")

    with tab_metars:
        with st.spinner("Fetching METARs from NOAA..."):
            metars_df = get_live_metars()
        if metars_df.empty:
            st.info("No METAR data available from NOAA Aviation Weather Center.")
        else:
            st.pydeck_chart(metar_station_map(metars_df))
            st.caption("METAR stations colored by flight category (VFR/MVFR/IFR/LIFR).")

    with tab_corridors:
        st.pydeck_chart(traffic_corridors_map(tf, airports=airports))
        st.caption("Simulated major air traffic corridors.")


if __name__ == "__main__":
    main()
