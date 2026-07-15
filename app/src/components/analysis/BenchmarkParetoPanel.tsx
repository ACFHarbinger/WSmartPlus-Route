/**
 * Single Pareto scatter panel for overflow vs profit (§G.1.2).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Download } from "lucide-react";
import { toast } from "sonner";
import { exportChartPng } from "../../utils/chartExport";
import { chartMetricDisplay, chartMetricUsesSymlog } from "../../utils/chartLogScale";
import { paretoFront, paretoStepLine } from "../../utils/pareto";
import {
  citySymbol,
  formatLogMeta,
  formatPolicyMeta,
  parsePolicyLabel,
  strategyColor,
} from "../../utils/simMetadata";
import type { ParetoPoint } from "../../utils/paretoPortfolio";

function fmt(n: number, d = 1) {
  return isFinite(n) ? n.toFixed(d) : "—";
}

function slugLabel(label: string) {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "pareto";
}

export function BenchmarkParetoPanel({
  label,
  points,
  logScale,
}: {
  label: string;
  points: ParetoPoint[];
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const option = useMemo(() => {
    const frontIds = new Set(paretoFront(points).map((p) => p.id));
    const step = paretoStepLine(paretoFront(points));
    const overflowSymlog = chartMetricUsesSymlog("overflows", logScale);
    const displayX = (x: number) => chartMetricDisplay(x, "profit", logScale) ?? x;
    const displayY = (y: number) => chartMetricDisplay(y, "overflows", logScale) ?? y;

    return {
      backgroundColor: "transparent",
      title: {
        text: `${label}${logScale ? " · symlog ovfl" : ""}`,
        left: "center",
        textStyle: { color: "#9090b0", fontSize: 10 },
      },
      grid: { left: 44, right: 10, top: 36, bottom: 32 },
      xAxis: {
        type: (logScale ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: logScale ? "Profit (log)" : "Profit",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
        minorSplitLine: { show: false },
      },
      yAxis: {
        type: (logScale && !overflowSymlog ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: overflowSymlog ? "Ovfl (symlog)" : logScale ? "Ovfl (log)" : "Ovfl",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "scatter",
          data: points.map((pt) => {
            const policyMeta = parsePolicyLabel(pt.policy);
            const onFront = frontIds.has(pt.id);
            return {
              name: pt.id,
              value: [displayX(pt.x), displayY(pt.y)],
              itemStyle: {
                color: onFront ? strategyColor(pt.policy, { [pt.policy]: policyMeta }) : "#6b7280",
              },
              symbol: citySymbol(pt.logMeta),
              symbolSize: onFront ? 9 : 6,
            };
          }),
          tooltip: {
            formatter: (p: { name: string; value: [number, number] }) => {
              const pt = points.find((x) => x.id === p.name);
              if (!pt) return p.name;
              const meta = parsePolicyLabel(pt.policy);
              const lines = [
                pt.policy,
                formatLogMeta(pt.logMeta),
                formatPolicyMeta(meta),
                `Profit: ${fmt(p.value[0])} €`,
                `Overflows: ${fmt(pt.y)}`,
                frontIds.has(pt.id) ? "Pareto-optimal" : "",
              ].filter(Boolean);
              return lines.join("<br/>");
            },
          },
        },
        ...(step.length > 1
          ? [
              {
                type: "line",
                data: step.map(([x, y]) => [displayX(x), displayY(y)]),
                lineStyle: { color: "#f3f4f6", type: "dashed", width: 1 },
                symbol: "none",
                tooltip: { show: false },
                z: 1,
              },
            ]
          : []),
      ],
    };
  }, [label, points, logScale]);

  if (points.length === 0) {
    return (
      <div className="card flex items-center justify-center h-[200px] text-xs text-canvas-muted">
        {label} — no runs
      </div>
    );
  }

  const exportStem = `pareto-${slugLabel(label)}`;

  return (
    <div className="card space-y-1">
      <div className="flex items-center justify-end">
        <button
          type="button"
          onClick={() => {
            if (exportChartPng({ current: chartRef.current }, `${exportStem}.png`)) {
              toast.success("Chart exported", { description: `${exportStem}.png` });
            } else {
              toast.error("Export failed", { description: "Chart is not ready" });
            }
          }}
          className="btn-ghost text-xs flex items-center gap-1"
          title="Export Pareto panel as PNG"
        >
          <Download size={11} />
          PNG
        </button>
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}
