/**
 * Single Pareto scatter panel for overflow vs profit (§G.1.2).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { paretoFront, paretoStepLine } from "../../utils/pareto";
import { parsePolicyLabel, strategyColor } from "../../utils/simMetadata";
import type { ParetoPoint } from "../../utils/paretoPortfolio";

function fmt(n: number, d = 1) {
  return isFinite(n) ? n.toFixed(d) : "—";
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
    const displayY = (y: number) => (logScale ? Math.max(y, 0.001) : y);

    return {
      backgroundColor: "transparent",
      title: { text: label, left: "center", textStyle: { color: "#9090b0", fontSize: 10 } },
      grid: { left: 44, right: 10, top: 36, bottom: 32 },
      xAxis: {
        type: "value",
        name: "Profit",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      },
      yAxis: {
        type: logScale ? "log" : "value",
        logBase: 10,
        name: "Ovfl",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "scatter",
          data: points.map((pt) => ({
            name: pt.policy,
            value: [pt.x, displayY(pt.y)],
            itemStyle: {
              color: strategyColor(pt.policy, { [pt.policy]: parsePolicyLabel(pt.policy) }),
            },
            symbolSize: frontIds.has(pt.id) ? 9 : 6,
          })),
          tooltip: {
            formatter: (p: { name: string; value: [number, number] }) =>
              `${p.name}<br/>Profit: ${fmt(p.value[0])} €<br/>Overflows: ${fmt(points.find((x) => x.policy === p.name)?.y ?? p.value[1])}`,
          },
        },
        ...(step.length > 1
          ? [
              {
                type: "line",
                data: step.map(([x, y]) => [x, displayY(y)]),
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

  return (
    <div className="card">
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}
