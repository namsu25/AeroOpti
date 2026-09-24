import type { RouteResult } from "../types";

function pctDelta(base: number, best: number): { label: string; positive: boolean } {
  if (Math.abs(base) < 1e-9) return { label: "n/a", positive: true };
  const pct = ((base - best) / base) * 100;
  return { label: `${pct >= 0 ? "▲" : "▼"} ${Math.abs(pct).toFixed(1)}%`, positive: pct >= 0 };
}

export default function KpiCards({ baseline, best }: { baseline: RouteResult; best: RouteResult }) {
  const cards = [
    { label: "Cost saving", value: `$${Math.round(baseline.cost_usd - best.cost_usd).toLocaleString()}`,
      delta: pctDelta(baseline.cost_usd, best.cost_usd) },
    { label: "Fuel saving", value: `${Math.round(baseline.fuel_kg - best.fuel_kg).toLocaleString()} kg`,
      delta: pctDelta(baseline.fuel_kg, best.fuel_kg) },
    { label: "CO2 avoided", value: `${Math.round(baseline.co2_kg - best.co2_kg).toLocaleString()} kg`,
      delta: pctDelta(baseline.co2_kg, best.co2_kg) },
    { label: "Time delta", value: `${((best.time_h - baseline.time_h) * 60).toFixed(0)} min`, delta: null },
  ];
  return (
    <div className="kpi-row">
      {cards.map((c) => (
        <div className="kpi-card" key={c.label}>
          <span className="kpi-label">{c.label}</span>
          <span className="kpi-value">{c.value}</span>
          {c.delta && (
            <span className="kpi-delta" style={c.delta.positive ? undefined : { color: "#f2545b", background: "rgba(242,84,91,0.1)" }}>
              {c.delta.label}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
