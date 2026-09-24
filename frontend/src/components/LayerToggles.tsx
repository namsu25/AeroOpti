import type { LayerKey } from "../types";

const LAYERS: { key: LayerKey; label: string; needsLive?: boolean }[] = [
  { key: "risk", label: "Weather risk" },
  { key: "wind", label: "Wind field" },
  { key: "corridors", label: "Traffic corridors" },
  { key: "live", label: "Live traffic", needsLive: true },
  { key: "metars", label: "NOAA METARs", needsLive: true },
];

interface Props {
  active: Set<LayerKey>;
  liveData: boolean;
  onToggle: (key: LayerKey) => void;
}

export default function LayerToggles({ active, liveData, onToggle }: Props) {
  return (
    <div className="layer-toggles">
      {LAYERS.map((l) => {
        const disabled = !!l.needsLive && !liveData;
        const isActive = active.has(l.key);
        return (
          <button
            key={l.key}
            type="button"
            className={`chip ${isActive ? "chip-active" : ""} ${disabled ? "chip-disabled" : ""}`}
            disabled={disabled}
            aria-pressed={isActive}
            onClick={() => onToggle(l.key)}
          >
            {l.label}
          </button>
        );
      })}
    </div>
  );
}
