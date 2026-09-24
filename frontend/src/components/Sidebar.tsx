import type { Config, Weights } from "../types";

interface Props {
  config: Config;
  origin: string;
  destination: string;
  aircraft: string;
  payloadPct: number;
  weights: Weights;
  liveData: boolean;
  loading: boolean;
  onChange: (patch: Partial<{
    origin: string; destination: string; aircraft: string;
    payloadPct: number; weights: Weights; liveData: boolean;
  }>) => void;
  onOptimize: () => void;
}

const PRESETS: Record<string, Weights> = {
  "Fuel-First": { fuel: 3.0, time: 0.3, risk: 1.0, airspace_fee: 0.3, traffic: 0.2 },
  "Time-Sensitive": { fuel: 0.5, time: 3.0, risk: 0.5, airspace_fee: 0.3, traffic: 0.5 },
  "Low-Risk": { fuel: 1.0, time: 0.5, risk: 5.0, airspace_fee: 0.3, traffic: 1.0 },
};

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label className="field slider-row">
      <span>{label}<b>{value.toFixed(1)}</b></span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function Sidebar({
  config, origin, destination, aircraft, payloadPct, weights, liveData, loading, onChange, onOptimize,
}: Props) {
  const airportCodes = Object.keys(config.airports).sort();
  const aircraftCodes = Object.keys(config.aircraft).sort();

  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <p className="eyebrow">Route configuration</p>

        <label className="field">
          Origin
          <select value={origin} onChange={(e) => onChange({ origin: e.target.value })}>
            {airportCodes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <label className="field">
          Destination
          <select value={destination} onChange={(e) => onChange({ destination: e.target.value })}>
            {airportCodes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <label className="field">
          Aircraft type
          <select value={aircraft} onChange={(e) => onChange({ aircraft: e.target.value })}>
            {aircraftCodes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>

        <Slider label="Payload %" value={payloadPct} min={50} max={100} step={1}
                onChange={(v) => onChange({ payloadPct: v })} />
      </div>

      <div className="sidebar-section">
        <p className="eyebrow">Scenario presets</p>
        <div className="preset-row">
          {Object.keys(PRESETS).map((name) => (
            <button key={name} className="preset-btn" onClick={() => onChange({ weights: PRESETS[name] })}>
              {name}
            </button>
          ))}
        </div>
      </div>

      <details className="weights-panel">
        <summary>Advanced — cost weights</summary>
        <Slider label="Fuel weight" value={weights.fuel} min={0.1} max={3} step={0.1}
                onChange={(v) => onChange({ weights: { ...weights, fuel: v } })} />
        <Slider label="Time weight" value={weights.time} min={0.1} max={3} step={0.1}
                onChange={(v) => onChange({ weights: { ...weights, time: v } })} />
        <Slider label="Risk weight" value={weights.risk} min={0.1} max={5} step={0.1}
                onChange={(v) => onChange({ weights: { ...weights, risk: v } })} />
        <Slider label="Airspace fee" value={weights.airspace_fee} min={0} max={1} step={0.1}
                onChange={(v) => onChange({ weights: { ...weights, airspace_fee: v } })} />
        <Slider label="Traffic avoidance" value={weights.traffic} min={0} max={3} step={0.1}
                onChange={(v) => onChange({ weights: { ...weights, traffic: v } })} />
      </details>

      <hr />

      <label className="field toggle-row">
        <input type="checkbox" checked={liveData} onChange={(e) => onChange({ liveData: e.target.checked })} />
        Enable live data
      </label>

      <button className="btn-primary" onClick={onOptimize} disabled={loading || origin === destination}>
        {loading ? "Optimizing…" : "Optimize routes"}
      </button>
      {origin === destination && <p className="error-text">Origin and destination must differ.</p>}
      {!config.model_ready && (
        <p className="error-text">
          Fuel model not trained. Run <code>python scripts/generate_sample_data.py</code> then{" "}
          <code>python scripts/train_models.py</code>.
        </p>
      )}
    </aside>
  );
}
