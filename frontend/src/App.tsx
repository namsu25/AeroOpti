import { useEffect, useState } from "react";
import { getConfig, getLiveMetars, getLiveTraffic, getTrafficField, getWeatherField, optimize } from "./api";
import Sidebar from "./components/Sidebar";
import KpiCards from "./components/KpiCards";
import RouteMap from "./components/RouteMap";
import LayerToggles from "./components/LayerToggles";
import { ComparisonBarChart, ParetoChart } from "./components/Charts";
import RouteTable from "./components/RouteTable";
import type {
  Config, LayerKey, LiveAircraft, Metar, OptimizeResponse, TrafficField, WeatherField, Weights,
} from "./types";
import "./App.css";

const DEFAULT_WEIGHTS: Weights = { fuel: 1.0, time: 0.5, risk: 2.0, airspace_fee: 0.3, traffic: 0.5 };

// Render-inspired palette applied client-side: neutral baseline, violet for the
// A* candidate, mint for the RL-refined candidate -- keeps presentation decisions
// out of the backend, which just returns routes in a fixed order.
const ROUTE_PALETTE = ["#8b8b95", "#9b52fb", "#34d399"];

function paletteize(res: OptimizeResponse): OptimizeResponse {
  return {
    ...res,
    routes: res.routes.map((r, i) => ({ ...r, color: ROUTE_PALETTE[i] ?? r.color })),
  };
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [weatherField, setWeatherField] = useState<WeatherField | null>(null);
  const [trafficField, setTrafficField] = useState<TrafficField | null>(null);
  const [liveAircraft, setLiveAircraft] = useState<LiveAircraft[]>([]);
  const [metars, setMetars] = useState<Metar[]>([]);

  const [origin, setOrigin] = useState("JFK");
  const [destination, setDestination] = useState("LHR");
  const [aircraft, setAircraft] = useState("B777");
  const [payloadPct, setPayloadPct] = useState(75);
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [liveData, setLiveData] = useState(false);
  const [activeLayers, setActiveLayers] = useState<Set<LayerKey>>(new Set(["corridors"]));

  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setError(String(e)));
    getWeatherField().then(setWeatherField).catch(() => {});
    getTrafficField().then(setTrafficField).catch(() => {});
  }, []);

  useEffect(() => {
    if (!liveData) {
      setLiveAircraft([]);
      setMetars([]);
      return;
    }
    let cancelled = false;
    const load = () => {
      getLiveTraffic().then((d) => !cancelled && setLiveAircraft(d)).catch(() => {});
      getLiveMetars().then((d) => !cancelled && setMetars(d)).catch(() => {});
    };
    load();
    const interval = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [liveData]);

  function handleSidebarChange(patch: Partial<{
    origin: string; destination: string; aircraft: string;
    payloadPct: number; weights: Weights; liveData: boolean;
  }>) {
    if (patch.origin !== undefined) setOrigin(patch.origin);
    if (patch.destination !== undefined) setDestination(patch.destination);
    if (patch.aircraft !== undefined) setAircraft(patch.aircraft);
    if (patch.payloadPct !== undefined) setPayloadPct(patch.payloadPct);
    if (patch.weights !== undefined) setWeights(patch.weights);
    if (patch.liveData !== undefined) setLiveData(patch.liveData);
  }

  function toggleLayer(key: LayerKey) {
    setActiveLayers((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }

  async function handleOptimize() {
    setLoading(true);
    setError(null);
    try {
      const res = await optimize(origin, destination, aircraft, payloadPct, weights);
      setResult(paletteize(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  if (!config) {
    return <div className="loading-screen">{error ?? "Loading AeroOpti…"}</div>;
  }

  const baseline = result?.routes[0];
  const best = result ? result.routes[result.best_index] : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-mark">A</span>
          <span className="topbar-wordmark">AeroOpti</span>
          <span className="topbar-tagline">AI flight route optimizer</span>
        </div>
        <div className="topbar-status">
          <span className="status-pill">
            <span className={`status-dot ${config.model_ready ? "on" : ""}`} />
            {config.model_ready ? "Model ready" : "Model not trained"}
          </span>
          <span className="status-pill">
            <span className={`status-dot ${liveData ? "live" : ""}`} />
            {liveData ? "Live data on" : "Live data off"}
          </span>
        </div>
      </header>

      <div className="shell">
        <Sidebar
          config={config} origin={origin} destination={destination} aircraft={aircraft}
          payloadPct={payloadPct} weights={weights} liveData={liveData} loading={loading}
          onChange={handleSidebarChange} onOptimize={handleOptimize}
        />
        <main className="main">
          <div className="page-header">
            <h1>Route optimizer</h1>
            <p>Compare a great-circle baseline against fuel/time-optimal and risk-aware candidates.</p>
          </div>

          {error && <div className="banner-error">{error}</div>}
          {!result && !error && (
            <div className="banner-info">Configure your route in the sidebar and click <b>Optimize Routes</b> to begin.</div>
          )}
          {baseline && best && <KpiCards baseline={baseline} best={best} />}

          <section className="map-section">
            <div className="map-card">
              <div className="map-header">
                <h2>Route map</h2>
                <LayerToggles active={activeLayers} liveData={liveData} onToggle={toggleLayer} />
              </div>
              <RouteMap
                config={config} origin={origin} destination={destination}
                routes={result?.routes ?? []} bestIndex={result?.best_index ?? -1}
                weatherField={weatherField} trafficField={trafficField}
                liveAircraft={liveAircraft} metars={metars} activeLayers={activeLayers}
              />
              <p className="map-caption">
                Solid = optimized routes, dashed = great-circle baseline. Toggle layers above to overlay live
                weather risk, wind vectors, simulated traffic corridors, live OpenSky traffic, and NOAA METARs
                on the same interactive map.
              </p>
            </div>
          </section>

          {result && (
            <>
              <section className="chart-row">
                <ParetoChart routes={result.routes} />
                <ComparisonBarChart routes={result.routes} />
              </section>
              <RouteTable routes={result.routes} />
            </>
          )}

          <footer>
            AeroOpti — Decision support only. Pilots and dispatchers retain final authority. Not for operational use.
          </footer>
        </main>
      </div>
    </div>
  );
}
