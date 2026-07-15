/**
 * Shared heatmap metric schema for §G.1.3 policy configuration heatmaps.
 */

import { displayBarValue } from "./chartLogScale";

export type HeatmapMode = "all" | "overflows" | "kg/km";

export const HEATMAP_METRICS = [
  { key: "profit", label: "Profit", higherBetter: true },
  { key: "kg/km", label: "kg/km", higherBetter: true },
  { key: "overflows", label: "Overflows", higherBetter: false },
  { key: "km", label: "km", higherBetter: false },
] as const;

export type HeatmapMetricKey = (typeof HEATMAP_METRICS)[number]["key"];

export function activeHeatmapMetrics(mode: HeatmapMode) {
  if (mode === "all") return [...HEATMAP_METRICS];
  return HEATMAP_METRICS.filter((m) => m.key === mode);
}

export function buildNormalizedHeatmapCells(
  policies: string[],
  metrics: Array<{ key: string; higherBetter: boolean }>,
  getRaw: (policy: string, metricKey: string) => number,
  policyOpacity?: (policy: string) => number,
  logScale = false
): { cells: Array<[number, number, number]>; raw: number[][] } {
  const raw: number[][] = [];
  const cells: Array<[number, number, number]> = [];

  for (let mi = 0; mi < metrics.length; mi++) {
    const { key, higherBetter } = metrics[mi];
    const row = policies.map((p) => {
      const v = getRaw(p, key);
      return logScale ? displayBarValue(v, key, true) : v;
    });
    raw.push(policies.map((p) => getRaw(p, key)));
    const min = Math.min(...row);
    const max = Math.max(...row);
    const span = max - min || 1;
    for (let pi = 0; pi < policies.length; pi++) {
      let norm = (row[pi] - min) / span;
      if (!higherBetter) norm = 1 - norm;
      const opacity = policyOpacity?.(policies[pi]) ?? 1;
      cells.push([pi, mi, norm * opacity]);
    }
  }

  return { cells, raw };
}

export const HEATMAP_VISUAL_MAP = {
  min: 0,
  max: 1,
  calculable: false,
  orient: "horizontal" as const,
  left: "center",
  bottom: 0,
  inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
  textStyle: { color: "#9090b0", fontSize: 9 },
  show: false,
};
