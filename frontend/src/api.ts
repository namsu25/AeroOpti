import type {
  Config,
  LiveAircraft,
  Metar,
  OptimizeResponse,
  TrafficField,
  WeatherField,
  Weights,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(detail.detail ?? `Request failed: ${resp.status}`);
  }
  return resp.json();
}

export function getConfig(): Promise<Config> {
  return getJSON("/api/config");
}

export function getWeatherField(): Promise<WeatherField> {
  return getJSON("/api/weather-field");
}

export function getTrafficField(): Promise<TrafficField> {
  return getJSON("/api/traffic-field");
}

export function getLiveTraffic(bbox?: [number, number, number, number]): Promise<LiveAircraft[]> {
  const qs = bbox ? `?min_lat=${bbox[0]}&min_lon=${bbox[1]}&max_lat=${bbox[2]}&max_lon=${bbox[3]}` : "";
  return getJSON(`/api/live/traffic${qs}`);
}

export function getLiveMetars(): Promise<Metar[]> {
  return getJSON("/api/live/metars");
}

export async function optimize(
  origin: string,
  destination: string,
  aircraft: string,
  payloadPct: number,
  weights: Weights,
): Promise<OptimizeResponse> {
  const resp = await fetch(`${API_BASE}/api/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      origin, destination, aircraft,
      payload_pct: payloadPct, weights,
    }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(detail.detail ?? `Optimize failed: ${resp.status}`);
  }
  return resp.json();
}
