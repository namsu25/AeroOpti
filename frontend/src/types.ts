export interface Weights {
  fuel: number;
  time: number;
  risk: number;
  airspace_fee: number;
  traffic: number;
}

export interface AircraftSpec {
  fuel_rate: number;
  payload_kg: number;
  cruise_alt_ft: number;
}

export interface Config {
  airports: Record<string, [number, number]>;
  aircraft: Record<string, AircraftSpec>;
  default_weights: Weights;
  model_ready: boolean;
}

export interface RouteResult {
  name: string;
  points: [number, number][];
  fuel_kg: number;
  time_h: number;
  risk: number;
  congestion: number;
  cost_usd: number;
  co2_kg: number;
  color: string;
}

export interface OptimizeResponse {
  routes: RouteResult[];
  best_index: number;
}

export interface WeatherField {
  lat_grid: number[];
  lon_grid: number[];
  risk: number[][];
  wind_u: number[][];
  wind_v: number[][];
  temperature: number[][];
}

export interface Corridor {
  origin: string;
  destination: string;
  intensity: number;
  path: [number, number][];
}

export interface TrafficField {
  lat_grid: number[];
  lon_grid: number[];
  density: number[][];
  corridors: Corridor[];
}

export interface LiveAircraft {
  icao24: string;
  callsign: string;
  origin_country: string;
  lat: number;
  lon: number;
  altitude_m: number | null;
  velocity_ms: number | null;
  track_deg: number | null;
  vertical_rate: number | null;
  on_ground: boolean;
}

export interface Metar {
  station: string;
  lat: number;
  lon: number;
  category: string;
  wind_kts: number;
  wind_dir: number;
  visibility_mi: number;
  ceiling_ft: number;
  temp_c: number;
  raw: string;
}

export type LayerKey = "risk" | "wind" | "corridors" | "live" | "metars";
