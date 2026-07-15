/**
 * Per-distribution policy heatmap facet for multi-run portfolios (§G.1.3).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Download } from "lucide-react";
import { exportChartPngWithToast, exportChartSvgWithToast } from "../../utils/chartExport";
import type { DayLogEntry } from "../../types";
import {
  activeHeatmapMetrics,
  buildNormalizedHeatmapCells,
  HEATMAP_VISUAL_MAP,
  type HeatmapMode,
} from "../../utils/heatmapMetrics";

export interface DistributionHeatmapRun {
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
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "distribution";
}

export function BenchmarkDistributionHeatmap({
  distributionLabel,
  runs,
  heatmapMode,
  logScale = false,
}: {
  distributionLabel: string;
  runs: DistributionHeatmapRun[];
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
          return `${policies[pi]}<br/>${metrics[mi].label}: ${fmt(raw[mi][pi], 2)}<br/>${runs.length} run(s)`;
        },
      },
    };
  }, [policies, runs, metrics, logScale]);

  const exportStem = `heatmap-distribution-${slugLabel(distributionLabel)}`;

  return (
    <div className="card space-y-1">
      <div className="flex items-center justify-between gap-1">
        <p className="text-[10px] text-canvas-muted font-mono">{distributionLabel}</p>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            type="button"
            onClick={() =>
              exportChartPngWithToast({ current: chartRef.current }, `${exportStem}.png`)
            }
            className="btn-ghost text-[10px] flex items-center gap-0.5"
            title="Export distribution heatmap as PNG"
          >
            <Download size={10} />
            PNG
          </button>
          <button
            type="button"
            onClick={() =>
              exportChartSvgWithToast({ current: chartRef.current }, `${exportStem}.svg`)
            }
            className="btn-ghost text-[10px] flex items-center gap-0.5"
            title="Export distribution heatmap as SVG"
          >
            <Download size={10} />
            SVG
          </button>
        </div>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(100, metrics.length * 36 + 40) }}
      />
    </div>
  );
}
