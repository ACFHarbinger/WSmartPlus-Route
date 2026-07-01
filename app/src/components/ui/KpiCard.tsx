import { TrendingDown, TrendingUp } from "lucide-react";

interface Props {
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  delta?: number | null;
  deltaLabel?: string;
  colorize?: boolean; // true = green when value is good (low for overflows, high for profit)
  lowerIsBetter?: boolean;
}

export function KpiCard({ label, value, unit, delta, colorize, lowerIsBetter }: Props) {
  const displayValue =
    value === null || value === undefined
      ? "—"
      : typeof value === "number"
      ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
      : value;

  const hasDelta = delta !== null && delta !== undefined;
  const isPositive = delta !== null && delta !== undefined && delta >= 0;
  const isGood = lowerIsBetter ? !isPositive : isPositive;

  return (
    <div className="kpi-card">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">
        {displayValue}
        {unit && <span className="text-sm font-normal text-canvas-muted ml-1">{unit}</span>}
      </span>
      {hasDelta && colorize && (
        <span className={isGood ? "kpi-delta-pos" : "kpi-delta-neg"}>
          <span className="inline-flex items-center gap-0.5">
            {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {delta! > 0 ? "+" : ""}
            {delta!.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
        </span>
      )}
    </div>
  );
}
