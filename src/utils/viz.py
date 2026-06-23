"""Plotly visualization helpers for routes, Pareto charts, weather, and traffic."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_route_map(routes: list[dict], origin: str, destination: str, airports: dict) -> go.Figure:
    """Create a Scattergeo map showing all route alternatives."""
    fig = go.Figure()
    for route in routes:
        lats = [p[0] for p in route["points"]]
        lons = [p[1] for p in route["points"]]
        dash = "dash" if "Baseline" in route["name"] else "solid"
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            name=route["name"],
            line=dict(width=2.5, color=route["color"], dash=dash),
            hovertemplate=f"{route['name']}<br>Fuel: {route['fuel_kg']:.0f} kg<br>Time: {route['time_h']:.2f} h",
        ))

    for code in [origin, destination]:
        lat, lon = airports[code]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers+text",
            marker=dict(size=10, color="#FF6B6B", symbol="circle"),
            text=[code], textposition="top center", textfont=dict(size=12, color="white"),
            showlegend=False,
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="rgb(30,30,30)",
        showocean=True, oceancolor="rgb(15,15,40)",
        showlakes=False, showcountries=True, countrycolor="rgb(60,60,60)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0), height=500,
        title="Route Comparison", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    )
    return fig


def plot_pareto(routes: list[dict]) -> go.Figure:
    """Scatter of fuel vs time with bubble size proportional to risk."""
    fig = go.Figure()
    for route in routes:
        fig.add_trace(go.Scatter(
            x=[route["time_h"]], y=[route["fuel_kg"]],
            mode="markers+text",
            marker=dict(size=max(route["risk"] * 50, 8), color=route["color"], opacity=0.8),
            text=[route["name"]], textposition="top center",
            name=route["name"],
        ))
    fig.update_layout(
        title="Trade-off Space",
        xaxis_title="Time (hours)", yaxis_title="Fuel (kg)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def plot_comparison_bars(routes: list[dict]) -> go.Figure:
    """Grouped bar chart comparing fuel, CO2, and cost across routes."""
    names = [r["name"] for r in routes]
    colors = [r["color"] for r in routes]

    fig = go.Figure()
    for i, route in enumerate(routes):
        fig.add_trace(go.Bar(
            name=route["name"],
            x=["Fuel (kg)", "CO₂ (kg)", "Cost ($)"],
            y=[route["fuel_kg"], route["co2_kg"], route["cost_usd"]],
            marker_color=route["color"],
        ))
    fig.update_layout(
        barmode="group", title="Route Metrics Comparison",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def plot_weather_heatmap(weather_field, routes: list[dict], lat_bounds: tuple, lon_bounds: tuple) -> go.Figure:
    """Heatmap of weather risk with route overlay."""
    lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])

    risk_sub = weather_field.risk[np.ix_(lat_mask, lon_mask)]
    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=risk_sub, x=lon_sub, y=lat_sub,
        colorscale="YlOrRd", zmin=0, zmax=1,
        colorbar=dict(title="Risk"),
    ))

    for route in routes:
        lats = [p[0] for p in route["points"]]
        lons = [p[1] for p in route["points"]]
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode="lines+markers",
            name=route["name"], line=dict(color=route["color"], width=2),
            marker=dict(size=3),
        ))

    fig.update_layout(
        title="Weather Risk Field",
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    return fig


def plot_wind_field(weather_field, lat_bounds: tuple, lon_bounds: tuple, routes: list[dict] | None = None) -> go.Figure:
    """Quiver-style wind vector field with speed heatmap background."""
    lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])

    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]
    u_sub = weather_field.wind_u[np.ix_(lat_mask, lon_mask)]
    v_sub = weather_field.wind_v[np.ix_(lat_mask, lon_mask)]
    speed = np.sqrt(u_sub ** 2 + v_sub ** 2)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=speed, x=lon_sub, y=lat_sub,
        colorscale="Blues", zmin=0,
        colorbar=dict(title="Wind (m/s)", x=1.02),
        name="Wind Speed",
    ))

    step = max(1, len(lat_sub) // 15)
    for i in range(0, len(lat_sub), step):
        for j in range(0, len(lon_sub), step):
            scale = 0.15
            fig.add_annotation(
                x=lon_sub[j], y=lat_sub[i],
                ax=lon_sub[j] + u_sub[i, j] * scale,
                ay=lat_sub[i] + v_sub[i, j] * scale,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=1.5, arrowcolor="white",
            )

    if routes:
        for route in routes:
            lats = [p[0] for p in route["points"]]
            lons = [p[1] for p in route["points"]]
            fig.add_trace(go.Scatter(
                x=lons, y=lats, mode="lines",
                name=route["name"], line=dict(color=route["color"], width=2.5),
            ))

    fig.update_layout(
        title="Wind Field (arrows = direction, color = speed)",
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    return fig


def plot_temperature_field(weather_field, lat_bounds: tuple, lon_bounds: tuple, routes: list[dict] | None = None) -> go.Figure:
    """Temperature deviation heatmap with route overlay."""
    lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])

    temp_sub = weather_field.temperature[np.ix_(lat_mask, lon_mask)]
    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=temp_sub, x=lon_sub, y=lat_sub,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Temp Dev (C)"),
        name="Temperature",
    ))

    if routes:
        for route in routes:
            lats = [p[0] for p in route["points"]]
            lons = [p[1] for p in route["points"]]
            fig.add_trace(go.Scatter(
                x=lons, y=lats, mode="lines",
                name=route["name"], line=dict(color=route["color"], width=2.5),
            ))

    fig.update_layout(
        title="ISA Temperature Deviation",
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    return fig


def plot_traffic_density(traffic_field, lat_bounds: tuple, lon_bounds: tuple, routes: list[dict] | None = None) -> go.Figure:
    """Traffic density heatmap with corridor lines and route overlay."""
    lat_mask = (traffic_field.lat_grid >= lat_bounds[0]) & (traffic_field.lat_grid <= lat_bounds[1])
    lon_mask = (traffic_field.lon_grid >= lon_bounds[0]) & (traffic_field.lon_grid <= lon_bounds[1])

    dens_sub = traffic_field.density[np.ix_(lat_mask, lon_mask)]
    lat_sub = traffic_field.lat_grid[lat_mask]
    lon_sub = traffic_field.lon_grid[lon_mask]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=dens_sub, x=lon_sub, y=lat_sub,
        colorscale="Viridis", zmin=0,
        colorbar=dict(title="Flights/h"),
        name="Traffic Density",
    ))

    for corr in traffic_field.corridors:
        path = corr["path"]
        clats = [p[0] for p in path]
        clons = [p[1] for p in path]
        in_view = any(
            lat_bounds[0] <= la <= lat_bounds[1] and lon_bounds[0] <= lo <= lon_bounds[1]
            for la, lo in zip(clats, clons)
        )
        if in_view:
            fig.add_trace(go.Scatter(
                x=clons, y=clats, mode="lines",
                line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
                name=f"{corr['origin']}-{corr['destination']}",
                showlegend=False,
                hovertemplate=f"{corr['origin']}-{corr['destination']}: {corr['intensity']:.0f} flights/h",
            ))

    if routes:
        for route in routes:
            lats = [p[0] for p in route["points"]]
            lons = [p[1] for p in route["points"]]
            fig.add_trace(go.Scatter(
                x=lons, y=lats, mode="lines+markers",
                name=route["name"], line=dict(color=route["color"], width=2.5),
                marker=dict(size=3),
            ))

    fig.update_layout(
        title="Air Traffic Density (flights/hour per cell)",
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    return fig


def plot_traffic_geo(traffic_field, routes: list[dict] | None = None, airports: dict | None = None) -> go.Figure:
    """Scattergeo map showing major traffic corridors with intensity-scaled width."""
    fig = go.Figure()

    max_intensity = max(c["intensity"] for c in traffic_field.corridors)
    for corr in traffic_field.corridors:
        path = corr["path"]
        lats = [p[0] for p in path]
        lons = [p[1] for p in path]
        width = 1 + 3 * (corr["intensity"] / max_intensity)
        opacity = 0.3 + 0.5 * (corr["intensity"] / max_intensity)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=width, color=f"rgba(255,165,0,{opacity})"),
            name=f"{corr['origin']}-{corr['destination']}",
            hovertemplate=f"{corr['origin']}-{corr['destination']}<br>{corr['intensity']:.0f} flights/h",
            showlegend=False,
        ))

    if routes:
        for route in routes:
            rlats = [p[0] for p in route["points"]]
            rlons = [p[1] for p in route["points"]]
            dash = "dash" if "Baseline" in route["name"] else "solid"
            fig.add_trace(go.Scattergeo(
                lat=rlats, lon=rlons, mode="lines",
                name=route["name"],
                line=dict(width=3, color=route["color"], dash=dash),
            ))

    if airports:
        for code, (lat, lon) in airports.items():
            fig.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], mode="markers+text",
                marker=dict(size=7, color="#FF6B6B"),
                text=[code], textposition="top center",
                textfont=dict(size=10, color="white"),
                showlegend=False,
            ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="rgb(30,30,30)",
        showocean=True, oceancolor="rgb(15,15,40)",
        showcountries=True, countrycolor="rgb(60,60,60)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0), height=500,
        title="Global Air Traffic Corridors",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    )
    return fig


def plot_weather_dashboard(weather_field, lat_bounds: tuple, lon_bounds: tuple, routes: list[dict] | None = None) -> go.Figure:
    """Combined 2x2 weather dashboard: risk, wind speed, temperature, wind vectors."""
    lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])

    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]
    risk_sub = weather_field.risk[np.ix_(lat_mask, lon_mask)]
    u_sub = weather_field.wind_u[np.ix_(lat_mask, lon_mask)]
    v_sub = weather_field.wind_v[np.ix_(lat_mask, lon_mask)]
    speed_sub = np.sqrt(u_sub ** 2 + v_sub ** 2)
    temp_sub = weather_field.temperature[np.ix_(lat_mask, lon_mask)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Turbulence Risk", "Wind Speed (m/s)", "Temperature Dev (C)", "Jet Stream Profile"),
        horizontal_spacing=0.08, vertical_spacing=0.1,
    )

    fig.add_trace(go.Heatmap(z=risk_sub, x=lon_sub, y=lat_sub, colorscale="YlOrRd", zmin=0, zmax=1,
                              colorbar=dict(title="Risk", x=0.45, y=0.8, len=0.35)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=speed_sub, x=lon_sub, y=lat_sub, colorscale="Blues", zmin=0,
                              colorbar=dict(title="m/s", x=1.0, y=0.8, len=0.35)), row=1, col=2)
    fig.add_trace(go.Heatmap(z=temp_sub, x=lon_sub, y=lat_sub, colorscale="RdBu_r", zmid=0,
                              colorbar=dict(title="C", x=0.45, y=0.2, len=0.35)), row=2, col=1)

    mid_lon_idx = len(lon_sub) // 2
    jet_profile = speed_sub[:, mid_lon_idx]
    fig.add_trace(go.Scatter(x=jet_profile, y=lat_sub, mode="lines",
                              line=dict(color="#4A90D9", width=2), name="Jet Stream"), row=2, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Latitude", row=2, col=2)

    if routes:
        for i_row, i_col in [(1, 1), (1, 2), (2, 1)]:
            for route in routes:
                lats = [p[0] for p in route["points"]]
                lons = [p[1] for p in route["points"]]
                fig.add_trace(go.Scatter(
                    x=lons, y=lats, mode="lines",
                    line=dict(color=route["color"], width=1.5),
                    showlegend=(i_row == 1 and i_col == 1),
                    name=route["name"],
                ), row=i_row, col=i_col)

    fig.update_layout(
        height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Weather Overview",
    )
    return fig
