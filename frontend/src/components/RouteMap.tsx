import { GeoJSONSource, LngLatBounds, Map as MlMap, NavigationControl, Popup } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";
import type {
  Config, LayerKey, LiveAircraft, Metar, RouteResult, TrafficField, WeatherField,
} from "../types";

// MapLibre's own demo style: a keyless, general-purpose-use vector basemap.
// (Raw tile.openstreetmap.org raster tiles are explicitly NOT meant for direct
// app use -- OSM's tile usage policy rate-limits/blocks bulk or automated
// fetching, which made the map silently stop loading under repeated reloads.)
const BASE_STYLE_URL = "https://demotiles.maplibre.org/style.json";

const CATEGORY_COLOR: Record<string, string> = {
  VFR: "#2ECC71", MVFR: "#4A90D9", IFR: "#E74C3C", LIFR: "#9B59B6",
};

function emptyFC(): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features: [] };
}

interface Props {
  config: Config | null;
  origin: string;
  destination: string;
  routes: RouteResult[];
  bestIndex: number;
  weatherField: WeatherField | null;
  trafficField: TrafficField | null;
  liveAircraft: LiveAircraft[];
  metars: Metar[];
  activeLayers: Set<LayerKey>;
}

export default function RouteMap({
  config, origin, destination, routes, bestIndex,
  weatherField, trafficField, liveAircraft, metars, activeLayers,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const popupRef = useRef<Popup | null>(null);
  const readyRef = useRef(false);

  // --- create the map once ---
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new MlMap({
      container: containerRef.current,
      style: BASE_STYLE_URL,
      center: [-40, 45],
      zoom: 2.2,
    });
    map.addControl(new NavigationControl(), "top-right");
    // Approximate the original "dark OSM" basemap look without needing a styled tile server / API key.
    map.getContainer().style.filter = "invert(1) hue-rotate(180deg) brightness(0.95) contrast(0.9)";
    popupRef.current = new Popup({ closeButton: false, closeOnClick: false });
    map.on("error", (e) => console.error("[AeroOpti map error]", e.error ?? e));
    // "style.load" fires once the style/sprite/glyphs are parsed and does NOT wait
    // for the basemap's own tiles to finish downloading -- unlike "load", which can
    // hang indefinitely if the basemap's tile server is slow, rate-limited, or
    // unreachable. Our own layers are all local GeoJSON pushed via setData, so
    // they have no reason to be gated on the basemap's tile network at all.
    map.on("style.load", () => {
      map.addSource("routes", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "routes-line", type: "line", source: "routes",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": ["get", "color"],
          "line-width": ["case", ["get", "best"], 5, 2.5],
          "line-dasharray": ["case", ["get", "baseline"], ["literal", [2, 2]], ["literal", [1, 0]]],
        },
      });

      map.addSource("airports", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "airports-point", type: "circle", source: "airports",
        paint: { "circle-radius": 6, "circle-color": "#FF6B6B", "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 },
      });
      map.addLayer({
        id: "airports-label", type: "symbol", source: "airports",
        layout: { "text-field": ["get", "code"], "text-offset": [0, 1.2], "text-size": 12 },
        paint: { "text-color": "#fff", "text-halo-color": "#000", "text-halo-width": 1 },
      });

      map.addSource("risk-heat", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "risk-heat-layer", type: "heatmap", source: "risk-heat",
        paint: {
          "heatmap-weight": ["get", "risk"],
          "heatmap-intensity": 1.2,
          "heatmap-radius": 18,
          "heatmap-opacity": 0.65,
          "heatmap-color": [
            "interpolate", ["linear"], ["heatmap-density"],
            0, "rgba(0,0,0,0)", 0.2, "#2ECC71", 0.5, "#F1C40F", 0.8, "#E67E22", 1, "#E74C3C",
          ],
        },
      });

      map.addSource("wind", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "wind-line", type: "line", source: "wind",
        paint: { "line-color": ["get", "color"], "line-width": 1.5, "line-opacity": 0.8 },
      });

      map.addSource("corridors", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "corridors-line", type: "line", source: "corridors",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": "#F5A623", "line-width": ["get", "width"], "line-opacity": 0.5 },
      });

      map.addSource("live", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "live-point", type: "circle", source: "live",
        paint: { "circle-radius": 4, "circle-color": "#5DADE2", "circle-stroke-color": "#0b2f4a", "circle-stroke-width": 1 },
      });

      map.addSource("metars", { type: "geojson", data: emptyFC() });
      map.addLayer({
        id: "metars-point", type: "circle", source: "metars",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "wind_kts"], 0, 5, 40, 14],
          "circle-color": ["get", "color"],
          "circle-stroke-color": "#fff", "circle-stroke-width": 1,
        },
      });

      for (const [layerId, html] of [
        ["routes-line", (p: any) => `<b>${p.name}</b><br/>Fuel: ${Math.round(p.fuel_kg)} kg<br/>Time: ${Number(p.time_h).toFixed(2)} h`],
        ["live-point", (p: any) => `<b>${p.callsign || p.icao24}</b><br/>Alt: ${Math.round(p.altitude_m ?? 0)} m<br/>Speed: ${Math.round(p.velocity_ms ?? 0)} m/s`],
        ["metars-point", (p: any) => `<b>${p.station}</b> (${p.category})<br/>${p.raw}`],
      ] as const) {
        map.on("mousemove", layerId, (e) => {
          map.getCanvas().style.cursor = "pointer";
          const f = e.features?.[0];
          if (!f || !popupRef.current) return;
          popupRef.current
            .setLngLat(e.lngLat)
            .setHTML(html(f.properties))
            .addTo(map);
        });
        map.on("mouseleave", layerId, () => {
          map.getCanvas().style.cursor = "";
          popupRef.current?.remove();
        });
      }

      readyRef.current = true;
    });

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      readyRef.current = false;
    };
  }, []);

  // --- routes + airports ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current || !config) return;

    const routeFeatures: GeoJSON.Feature[] = routes.map((r, i) => ({
      type: "Feature",
      properties: {
        name: r.name, fuel_kg: r.fuel_kg, time_h: r.time_h, color: r.color,
        best: i === bestIndex, baseline: i === 0,
      },
      geometry: { type: "LineString", coordinates: r.points.map(([lat, lon]) => [lon, lat]) },
    }));
    (map.getSource("routes") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features: routeFeatures,
    });

    const codes = new Set([origin, destination]);
    const airportFeatures: GeoJSON.Feature[] = [...codes].filter((c) => config.airports[c]).map((code) => {
      const [lat, lon] = config.airports[code];
      return {
        type: "Feature", properties: { code },
        geometry: { type: "Point", coordinates: [lon, lat] },
      };
    });
    (map.getSource("airports") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features: airportFeatures,
    });

    if (routes.length > 0) {
      const bounds = new LngLatBounds();
      routes.forEach((r) => r.points.forEach(([lat, lon]) => bounds.extend([lon, lat])));
      map.fitBounds(bounds, { padding: 60, maxZoom: 6, duration: 500 });
    } else if (airportFeatures.length === 2) {
      const bounds = new LngLatBounds();
      airportFeatures.forEach((f) => bounds.extend((f.geometry as GeoJSON.Point).coordinates as [number, number]));
      map.fitBounds(bounds, { padding: 80, maxZoom: 5, duration: 500 });
    }
  }, [routes, bestIndex, origin, destination, config]);

  // --- weather risk heatmap + wind field ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;

    const riskFeatures: GeoJSON.Feature[] = [];
    const windFeatures: GeoJSON.Feature[] = [];
    if (weatherField) {
      const { lat_grid, lon_grid, risk, wind_u, wind_v } = weatherField;
      for (let i = 0; i < lat_grid.length; i += 1) {
        for (let j = 0; j < lon_grid.length; j += 1) {
          const lat = lat_grid[i];
          const lon = lon_grid[j];
          const r = risk[i][j];
          if (r > 0.05) {
            riskFeatures.push({
              type: "Feature", properties: { risk: r },
              geometry: { type: "Point", coordinates: [lon, lat] },
            });
          }
          if (i % 4 === 0 && j % 4 === 0) {
            const u = wind_u[i][j];
            const v = wind_v[i][j];
            const speed = Math.sqrt(u * u + v * v);
            if (speed < 1) continue;
            const scale = 0.35;
            const dLon = (u / speed) * scale * speed * 0.15;
            const dLat = (v / speed) * scale * speed * 0.15;
            windFeatures.push({
              type: "Feature",
              properties: { color: speed > 25 ? "#E74C3C" : speed > 12 ? "#F5A623" : "#4A90D9" },
              geometry: { type: "LineString", coordinates: [[lon, lat], [lon + dLon, lat + dLat]] },
            });
          }
        }
      }
    }
    (map.getSource("risk-heat") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features: riskFeatures,
    });
    (map.getSource("wind") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features: windFeatures,
    });
  }, [weatherField]);

  // --- traffic corridors ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const features: GeoJSON.Feature[] = (trafficField?.corridors ?? []).map((c) => ({
      type: "Feature",
      properties: { width: Math.max(1, c.intensity / 4) },
      geometry: { type: "LineString", coordinates: c.path.map(([lat, lon]) => [lon, lat]) },
    }));
    (map.getSource("corridors") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features,
    });
  }, [trafficField]);

  // --- live traffic ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const features: GeoJSON.Feature[] = liveAircraft
      .filter((a) => Number.isFinite(a.lat) && Number.isFinite(a.lon))
      .map((a) => ({
        type: "Feature",
        properties: { icao24: a.icao24, callsign: a.callsign, altitude_m: a.altitude_m, velocity_ms: a.velocity_ms },
        geometry: { type: "Point", coordinates: [a.lon, a.lat] },
      }));
    (map.getSource("live") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features,
    });
  }, [liveAircraft]);

  // --- METARs ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const features: GeoJSON.Feature[] = metars.map((m) => ({
      type: "Feature",
      properties: { ...m, color: CATEGORY_COLOR[m.category] ?? "#95A5A6" },
      geometry: { type: "Point", coordinates: [m.lon, m.lat] },
    }));
    (map.getSource("metars") as GeoJSONSource | undefined)?.setData({
      type: "FeatureCollection", features,
    });
  }, [metars]);

  // --- layer visibility toggles ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const vis = (id: string, on: boolean) => map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    vis("risk-heat-layer", activeLayers.has("risk"));
    vis("wind-line", activeLayers.has("wind"));
    vis("corridors-line", activeLayers.has("corridors"));
    vis("live-point", activeLayers.has("live"));
    vis("metars-point", activeLayers.has("metars"));
  }, [activeLayers]);

  return <div ref={containerRef} className="route-map" />;
}
