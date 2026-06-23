"""MapLibre + pydeck map visualizations using OpenStreetMap tiles."""

import numpy as np
import pandas as pd
import pydeck as pdk

OSM_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
OSM_POSITRON_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
OSM_VOYAGER_STYLE = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"


def _hex_to_rgba(hex_color: str, alpha: int = 220) -> list[int]:
    """Convert #RRGGBB to [R, G, B, A]."""
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def _route_center(routes: list[dict], airports: dict | None = None, origin: str = "", dest: str = "") -> tuple[float, float, float]:
    """Compute map center lat/lon and zoom from route points."""
    all_lats, all_lons = [], []
    for r in routes:
        for p in r["points"]:
            all_lats.append(p[0])
            all_lons.append(p[1])
    if not all_lats:
        return (40.0, -30.0, 3)
    center_lat = (min(all_lats) + max(all_lats)) / 2
    center_lon = (min(all_lons) + max(all_lons)) / 2
    span = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons))
    zoom = max(1, min(8, 7 - np.log2(max(span, 1))))
    return (center_lat, center_lon, zoom)


def route_map(
    routes: list[dict],
    origin: str,
    destination: str,
    airports: dict,
    height: int = 550,
) -> pdk.Deck:
    """Render route alternatives on a MapLibre dark basemap with arc and path layers."""
    layers = []
    center_lat, center_lon, zoom = _route_center(routes, airports, origin, destination)

    for route in routes:
        color = _hex_to_rgba(route["color"])
        path_data = [{
            "path": [[p[1], p[0]] for p in route["points"]],
            "name": route["name"],
            "fuel": route["fuel_kg"],
            "time": route["time_h"],
        }]
        width = 2 if "Baseline" in route["name"] else 4
        dash = [6, 4] if "Baseline" in route["name"] else None

        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=color,
            width_min_pixels=width,
            get_width=width,
            pickable=True,
        ))

    airport_data = []
    for code in set([origin, destination]):
        lat, lon = airports[code]
        airport_data.append({"code": code, "lat": lat, "lon": lon})

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_radius=40000,
        get_fill_color=[255, 107, 107, 230],
        pickable=True,
    ))
    layers.append(pdk.Layer(
        "TextLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_text="code",
        get_size=16,
        get_color=[255, 255, 255, 255],
        get_alignment_baseline="'bottom'",
    ))

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0,
        ),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
        tooltip={"text": "{name}\nFuel: {fuel} kg\nTime: {time} h"},
    )


def live_traffic_map(
    traffic_df: pd.DataFrame,
    routes: list[dict] | None = None,
    airports: dict | None = None,
    height: int = 550,
) -> pdk.Deck:
    """Render live aircraft positions from OpenSky as a scatter layer on MapLibre."""
    layers = []

    if not traffic_df.empty:
        df = traffic_df.copy()
        df["altitude_km"] = (df["altitude_m"].fillna(0) / 1000).clip(0, 15)
        df["speed_color_r"] = (df["velocity_ms"].fillna(0) / 300 * 255).clip(0, 255).astype(int)
        df["speed_color_g"] = (200 - df["velocity_ms"].fillna(0) / 300 * 100).clip(0, 255).astype(int)

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lon", "lat"],
            get_radius="altitude_km * 800 + 2000",
            get_fill_color=["speed_color_r", "speed_color_g", 100, 160],
            pickable=True,
            radius_min_pixels=2,
            radius_max_pixels=8,
        ))

    if routes:
        for route in routes:
            color = _hex_to_rgba(route["color"])
            path_data = [{"path": [[p[1], p[0]] for p in route["points"]], "name": route["name"]}]
            layers.append(pdk.Layer(
                "PathLayer", data=path_data, get_path="path",
                get_color=color, width_min_pixels=3, pickable=True,
            ))

    if airports:
        ap_data = [{"code": c, "lat": v[0], "lon": v[1]} for c, v in airports.items()]
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=ap_data,
            get_position=["lon", "lat"], get_radius=35000,
            get_fill_color=[255, 107, 107, 220], pickable=True,
        ))
        layers.append(pdk.Layer(
            "TextLayer", data=ap_data,
            get_position=["lon", "lat"], get_text="code",
            get_size=14, get_color=[255, 255, 255],
        ))

    center = (45.0, -20.0, 3) if routes is None else _route_center(routes)

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=center[2], pitch=0),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
        tooltip={"text": "{callsign}\n{origin_country}\nAlt: {altitude_m}m\nSpeed: {velocity_ms} m/s"},
    )


def weather_risk_map(
    weather_field,
    routes: list[dict] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    height: int = 500,
) -> pdk.Deck:
    """Heatmap of weather risk on a MapLibre basemap with route overlay."""
    if lat_bounds:
        lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    else:
        lat_mask = np.ones(len(weather_field.lat_grid), dtype=bool)
    if lon_bounds:
        lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])
    else:
        lon_mask = np.ones(len(weather_field.lon_grid), dtype=bool)

    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]
    risk_sub = weather_field.risk[np.ix_(lat_mask, lon_mask)]

    heat_data = []
    step = max(1, len(lat_sub) // 40)
    for i in range(0, len(lat_sub), step):
        for j in range(0, len(lon_sub), step):
            if risk_sub[i, j] > 0.05:
                heat_data.append({
                    "lat": float(lat_sub[i]),
                    "lon": float(lon_sub[j]),
                    "weight": float(risk_sub[i, j]),
                })

    layers = []
    if heat_data:
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position=["lon", "lat"],
            get_weight="weight",
            radiusPixels=30,
            intensity=1.5,
            threshold=0.05,
            color_range=[
                [255, 255, 178], [254, 204, 92], [253, 141, 60],
                [240, 59, 32], [189, 0, 38],
            ],
        ))

    if routes:
        for route in routes:
            color = _hex_to_rgba(route["color"])
            path_data = [{"path": [[p[1], p[0]] for p in route["points"]], "name": route["name"]}]
            layers.append(pdk.Layer(
                "PathLayer", data=path_data, get_path="path",
                get_color=color, width_min_pixels=3, pickable=True,
            ))

    center = (40, -30, 3) if not routes else _route_center(routes)

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=center[2]),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
    )


def wind_barb_map(
    weather_field,
    routes: list[dict] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    height: int = 500,
) -> pdk.Deck:
    """Wind field as colored arrows on MapLibre showing direction and speed."""
    if lat_bounds:
        lat_mask = (weather_field.lat_grid >= lat_bounds[0]) & (weather_field.lat_grid <= lat_bounds[1])
    else:
        lat_mask = np.ones(len(weather_field.lat_grid), dtype=bool)
    if lon_bounds:
        lon_mask = (weather_field.lon_grid >= lon_bounds[0]) & (weather_field.lon_grid <= lon_bounds[1])
    else:
        lon_mask = np.ones(len(weather_field.lon_grid), dtype=bool)

    lat_sub = weather_field.lat_grid[lat_mask]
    lon_sub = weather_field.lon_grid[lon_mask]
    u_sub = weather_field.wind_u[np.ix_(lat_mask, lon_mask)]
    v_sub = weather_field.wind_v[np.ix_(lat_mask, lon_mask)]

    arrow_data = []
    step = max(1, len(lat_sub) // 20)
    scale = 0.15
    for i in range(0, len(lat_sub), step):
        for j in range(0, len(lon_sub), step):
            speed = float(np.sqrt(u_sub[i, j] ** 2 + v_sub[i, j] ** 2))
            if speed < 1.0:
                continue
            src_lon = float(lon_sub[j])
            src_lat = float(lat_sub[i])
            dst_lon = src_lon + float(u_sub[i, j]) * scale
            dst_lat = src_lat + float(v_sub[i, j]) * scale
            r = min(255, int(speed * 8))
            g = min(255, int(200 - speed * 4))
            b = 255
            arrow_data.append({
                "src": [src_lon, src_lat],
                "dst": [dst_lon, dst_lat],
                "speed": round(speed, 1),
                "color": [r, g, b, 200],
            })

    layers = []
    if arrow_data:
        layers.append(pdk.Layer(
            "ArcLayer",
            data=arrow_data,
            get_source_position="src",
            get_target_position="dst",
            get_source_color="color",
            get_target_color="color",
            get_width=2,
            pickable=True,
        ))

    if routes:
        for route in routes:
            color = _hex_to_rgba(route["color"])
            path_data = [{"path": [[p[1], p[0]] for p in route["points"]], "name": route["name"]}]
            layers.append(pdk.Layer(
                "PathLayer", data=path_data, get_path="path",
                get_color=color, width_min_pixels=3, pickable=True,
            ))

    center = (40, -30, 3) if not routes else _route_center(routes)

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=center[2]),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
        tooltip={"text": "Wind: {speed} m/s"},
    )


def metar_station_map(
    metars_df: pd.DataFrame,
    routes: list[dict] | None = None,
    height: int = 500,
) -> pdk.Deck:
    """Plot METAR stations colored by flight category on MapLibre."""
    layers = []

    if not metars_df.empty:
        df = metars_df.copy()
        cat_colors = {"VFR": [0, 200, 0], "MVFR": [0, 100, 255], "IFR": [255, 0, 0], "LIFR": [200, 0, 200]}
        df["color"] = df["category"].map(lambda c: cat_colors.get(c, [150, 150, 150]))
        df["radius"] = df["wind_kts"].clip(1, 50) * 600 + 5000

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lon", "lat"],
            get_radius="radius",
            get_fill_color="color",
            opacity=0.6,
            pickable=True,
            radius_min_pixels=3,
            radius_max_pixels=12,
        ))

    if routes:
        for route in routes:
            color = _hex_to_rgba(route["color"])
            path_data = [{"path": [[p[1], p[0]] for p in route["points"]], "name": route["name"]}]
            layers.append(pdk.Layer(
                "PathLayer", data=path_data, get_path="path",
                get_color=color, width_min_pixels=3,
            ))

    center = (40, -30, 3) if not routes else _route_center(routes)

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=center[2]),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
        tooltip={"text": "{station}\n{category}\nWind: {wind_kts} kts @ {wind_dir}deg\nVis: {visibility_mi} mi\nCeiling: {ceiling_ft} ft"},
    )


def traffic_corridors_map(
    traffic_field,
    routes: list[dict] | None = None,
    airports: dict | None = None,
    height: int = 550,
) -> pdk.Deck:
    """Render simulated traffic corridors as arc layers on MapLibre."""
    layers = []

    if hasattr(traffic_field, "corridors"):
        max_intensity = max(c["intensity"] for c in traffic_field.corridors)
        arc_data = []
        for corr in traffic_field.corridors:
            path = corr["path"]
            src = path[0]
            dst = path[-1]
            norm = corr["intensity"] / max_intensity
            alpha = int(80 + 175 * norm)
            arc_data.append({
                "src": [src[1], src[0]],
                "dst": [dst[1], dst[0]],
                "intensity": corr["intensity"],
                "pair": f"{corr['origin']}-{corr['destination']}",
                "color": [255, 165, 0, alpha],
                "width": 1 + 5 * norm,
            })

        layers.append(pdk.Layer(
            "ArcLayer",
            data=arc_data,
            get_source_position="src",
            get_target_position="dst",
            get_source_color="color",
            get_target_color="color",
            get_width="width",
            pickable=True,
        ))

    if routes:
        for route in routes:
            color = _hex_to_rgba(route["color"])
            path_data = [{"path": [[p[1], p[0]] for p in route["points"]], "name": route["name"]}]
            layers.append(pdk.Layer(
                "PathLayer", data=path_data, get_path="path",
                get_color=color, width_min_pixels=3, pickable=True,
            ))

    if airports:
        ap_data = [{"code": c, "lat": v[0], "lon": v[1]} for c, v in airports.items()]
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=ap_data,
            get_position=["lon", "lat"], get_radius=35000,
            get_fill_color=[255, 107, 107, 220], pickable=True,
        ))
        layers.append(pdk.Layer(
            "TextLayer", data=ap_data,
            get_position=["lon", "lat"], get_text="code",
            get_size=14, get_color=[255, 255, 255],
        ))

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=40, longitude=-20, zoom=2.5, pitch=25),
        map_provider="maplibre",
        map_style=OSM_DARK_STYLE,
        height=height,
        tooltip={"text": "{pair}\n{intensity} flights/h"},
    )
