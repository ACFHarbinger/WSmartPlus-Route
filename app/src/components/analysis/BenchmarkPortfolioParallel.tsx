/**
 * Portfolio parallel coordinates — one polyline per loaded simulation log (§G.1.4).
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { cityScaleLabel, parseLogPath } from "../../utils/simMetadata";
import type { DayLogEntry } from "../../types";

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

export interface PortfolioParallelRun {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function BenchmarkPortfolioParallel({ runs }: { runs: PortfolioParallelRun[] }) {
  const lines = useMemo(
    () =>
      runs.map((run, i) => {
        const meta = parseLogPath(run.path);
        const vals = run.entries.map((e) => e.data as Record<string, number>);
        const profit = mean(vals.map((d) => d.profit).filter((v): v is number => v != null));
        const kgkm = mean(vals.map((d) => d["kg/km"]).filter((v): v is number => v != null));
        const overflows = mean(vals.map((d) => d.overflows).filter((v): v is number => v != null));
        const km = mean(vals.map((d) => d.km).filter((v): v is number => v != null));
        return {
          name: run.label,
          color: COLORS[i % COLORS.length],
          value: [meta.scale ?? 100, profit, kgkm, overflows, km],
          meta,
        };
      }),
    [runs]
  );

  const option = useMemo(() => {
    const schema = [
      { dim: 0, name: "N", max: Math.max(...lines.map((l) => l.value[0]), 1) * 1.1 },
      { dim: 1, name: "Profit", max: Math.max(...lines.map((l) => l.value[1]), 1) * 1.1 },
      { dim: 2, name: "kg/km", max: Math.max(...lines.map((l) => l.value[2]), 0.01) * 1.1 },
      { dim: 3, name: "Overflows", max: Math.max(...lines.map((l) => l.value[3]), 1) * 1.1 },
      { dim: 4, name: "km", max: Math.max(...lines.map((l) => l.value[4]), 1) * 1.1 },
    ];

    return {
      backgroundColor: "transparent",
      parallelAxis: schema.map((s) => ({
        dim: s.dim,
        name: s.name,
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLine: { lineStyle: { color: "#2d2d50" } },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      })),
      parallel: { left: 50, right: 24, top: 28, bottom: 36 },
      series: [
        {
          type: "parallel",
          lineStyle: { width: 1.5, opacity: 0.75 },
          data: lines.map((l) => ({
            name: l.name,
            value: l.value,
            lineStyle: { color: l.color },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const line = lines.find((l) => l.name === p.name);
          if (!line) return p.name;
          return [
            p.name,
            cityScaleLabel(line.meta),
            line.meta.distribution,
            line.meta.improver,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [lines]);

  return (
    <div className="card space-y-2">
      <p className="text-xs font-semibold text-gray-300">Portfolio Parallel Coordinates (§G.1.4)</p>
      <p className="text-[10px] text-canvas-muted">
        {runs.length} simulation log(s) — one polyline per loaded run
      </p>
      <ReactECharts option={option} style={{ height: Math.min(360, 120 + runs.length * 4) }} />
    </div>
  );
}
