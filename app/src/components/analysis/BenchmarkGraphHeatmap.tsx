/**
 * Per-graph policy heatmap facet for multi-run benchmarks (§G.1.3).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
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

function slugLabel(label: string) {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "graph";
}

export function BenchmarkGraphHeatmap({
  graphLabel,
  runs,
  heatmapMode,
  logScale = false,
}: {
  graphLabel: string;
  runs: GraphHeatmapRun[];
  heatmapMode: HeatmapMode;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
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

    const { cells, raw } = buildNormalizedHeatmapCells(
      policies,
      metrics,
      getRaw,
      undefined,
      logScale
    );

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
  }, [policies, runs, metrics, logScale]);

  const exportStem = `heatmap-graph-${slugLabel(graphLabel)}`;

  return (
    <div className="card space-y-1">
      <div className="flex items-center justify-between gap-1">
        <p className="text-[10px] text-canvas-muted font-mono">{graphLabel}</p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem={exportStem}
          size={10}
          className="shrink-0"
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(100, metrics.length * 36 + 40) }}
      />
    </div>
  );
}
