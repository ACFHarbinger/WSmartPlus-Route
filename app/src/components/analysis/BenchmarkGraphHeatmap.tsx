/**
 * Per-graph policy heatmap facet for multi-run benchmarks (§G.1.3).
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import type { DayLogEntry } from "../../types";
import {
  activeHeatmapMetrics,
  buildNormalizedHeatmapCells,
  HEATMAP_VISUAL_MAP,
  type HeatmapMode,
} from "../../utils/heatmapMetrics";

export interface GraphHeatmapRun {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function fmt(n: number, d = 1) {
  return isFinite(n) ? n.toFixed(d) : "—";
}

export function BenchmarkGraphHeatmap({
  graphLabel,
  runs,
  heatmapMode,
}: {
  graphLabel: string;
  runs: GraphHeatmapRun[];
  heatmapMode: HeatmapMode;
}) {
  const policies = useMemo(
    () => [...new Set(runs.flatMap((r) => r.entries.map((e) => e.policy)))],
    [runs]
  );

  const metrics = useMemo(() => activeHeatmapMetrics(heatmapMode), [heatmapMode]);

  const option = useMemo(() => {
    const getRaw = (policy: string, metricKey: string) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .filter((e) => e.policy === policy)
          .map((e) => (e.data as Record<string, number>)[metricKey])
          .filter((v): v is number => v != null)
      );
      return mean(vals);
    };

    const { cells, raw } = buildNormalizedHeatmapCells(policies, metrics, getRaw);

    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 16, top: 8, bottom: 40 },
      xAxis: {
        type: "category",
        data: policies,
        axisLabel: { color: "#9090b0", fontSize: 8, rotate: 25 },
      },
      yAxis: {
        type: "category",
        data: metrics.map((m) => m.label),
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: HEATMAP_VISUAL_MAP,
      series: [{ type: "heatmap", data: cells, label: { show: false } }],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const [pi, mi] = p.value;
          return `${policies[pi]}<br/>${metrics[mi].label}: ${fmt(raw[mi][pi], 2)}`;
        },
      },
    };
  }, [policies, runs, metrics]);

  return (
    <div className="card space-y-1">
      <p className="text-[10px] text-canvas-muted font-mono">{graphLabel}</p>
      <ReactECharts option={option} style={{ height: Math.max(100, metrics.length * 36 + 40) }} />
    </div>
  );
}
