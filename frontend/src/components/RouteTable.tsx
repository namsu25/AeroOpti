import type { RouteResult } from "../types";

export default function RouteTable({ routes }: { routes: RouteResult[] }) {
  return (
    <div className="table-card">
      <h3>Route details</h3>
      <table>
        <thead>
          <tr>
            <th>Route</th><th>Fuel (kg)</th><th>Time (h)</th><th>Risk (0-1)</th>
            <th>Congestion (0-1)</th><th>CO2 (kg)</th><th>Est. Cost ($)</th>
          </tr>
        </thead>
        <tbody>
          {routes.map((r) => (
            <tr key={r.name}>
              <td><span className="swatch" style={{ background: r.color }} />{r.name}</td>
              <td>{r.fuel_kg.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td>{r.time_h.toFixed(2)}</td>
              <td>{r.risk.toFixed(3)}</td>
              <td>{r.congestion.toFixed(3)}</td>
              <td>{r.co2_kg.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td>{r.cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
