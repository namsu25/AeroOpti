import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { RouteResult } from "../types";

const GRID = "#232328";
const AXIS = "#68686f";
const TOOLTIP_STYLE = {
  background: "#18181c", border: "1px solid rgba(255,255,255,0.16)", borderRadius: 8,
  fontSize: 12.5, fontFamily: "Inter, sans-serif",
};
const LEGEND_STYLE = { fontSize: 12.5, color: "#9a9aa4" };

export function ParetoChart({ routes }: { routes: RouteResult[] }) {
  return (
    <div className="chart-card">
      <h3>Trade-off space</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis type="number" dataKey="time_h" name="Time" unit="h" stroke={AXIS} tick={{ fontSize: 12 }} />
          <YAxis type="number" dataKey="fuel_kg" name="Fuel" unit="kg" stroke={AXIS} tick={{ fontSize: 12 }} />
          <ZAxis type="number" dataKey="risk" range={[80, 400]} name="Risk" />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "#68686f" }}
            contentStyle={TOOLTIP_STYLE}
            labelStyle={{ color: "#f4f4f5" }}
            formatter={(value, name) => {
              const n = Number(value);
              const label = name === "fuel_kg" ? `${Math.round(n)} kg` : name === "time_h" ? `${n.toFixed(2)} h` : value;
              return [label, name];
            }}
          />
          {routes.map((r) => (
            <Scatter key={r.name} name={r.name} data={[r]} fill={r.color} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ComparisonBarChart({ routes }: { routes: RouteResult[] }) {
  const metrics = [
    { key: "fuel_kg", label: "Fuel (kg)" },
    { key: "co2_kg", label: "CO₂ (kg)" },
    { key: "cost_usd", label: "Cost ($)" },
  ] as const;
  const data = metrics.map((m) => {
    const row: Record<string, number | string> = { metric: m.label };
    routes.forEach((r) => { row[r.name] = Math.round(r[m.key]); });
    return row;
  });

  return (
    <div className="chart-card">
      <h3>Route metrics comparison</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="metric" stroke={AXIS} tick={{ fontSize: 12 }} />
          <YAxis stroke={AXIS} tick={{ fontSize: 12 }} />
          <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={{ color: "#f4f4f5" }} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
          <Legend wrapperStyle={LEGEND_STYLE} />
          {routes.map((r) => (
            <Bar key={r.name} dataKey={r.name} fill={r.color} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
