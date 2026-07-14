/**
 * Per-distribution policy heatmap facet for multi-run portfolios (§G.1.3).
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import type { DayLogEntry } from "../../types";

const HEAT_METRICS = [
  { key: "overflows", label: "Overflows", higherBetter: false },
  { key: "kg/km", label: "kg/km", higherBetter: true },
] as const;

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

export function BenchmarkDistributionHeatmap({
  distributionLabel,
  runs,
  heatmapMode,
}: {
  distributionLabel: string;
  runs: DistributionHeatmapRun[];
  heatmapMode: "overflows" | "kg/km";
}) {
  const policies = useMemo(
    () => [...new Set(runs.flatMap((r) => r.entries.map((e) => e.policy)))],
    [runs]
  );

  const option = useMemo(() => {
    const metric = HEAT_METRICS.find((m) => m.key === heatmapMode) ?? HEAT_METRICS[0];
    const raw = policies.map((p) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[heatmapMode])
          .filter((v): v is number => v != null)
      );
      return mean(vals);
    });
    const min = Math.min(...raw);
    const max = Math.max(...raw);
    const span = max - min || 1;
    const cells: Array<[number, number, number]> = raw.map((v, pi) => {
      let norm = (v - min) / span;
      if (!metric.higherBetter) norm = 1 - norm;
      return [pi, 0, norm];
    });

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
        data: [metric.label],
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
        textStyle: { color: "#9090b0", fontSize: 9 },
        show: false,
      },
      series: [{ type: "heatmap", data: cells, label: { show: false } }],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const pi = p.value[0];
          return `${policies[pi]}<br/>${metric.label}: ${fmt(raw[pi], 2)}<br/>${runs.length} run(s)`;
        },
      },
    };
  }, [policies, runs, heatmapMode]);

  return (
    <div className="card space-y-1">
      <p className="text-[10px] text-canvas-muted font-mono">{distributionLabel}</p>
      <ReactECharts option={option} style={{ height: 100 }} />
    </div>
  );
}
