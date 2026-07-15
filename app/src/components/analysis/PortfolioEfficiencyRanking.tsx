/**
 * Rank run×policy configurations by mean kg/km across a loaded portfolio (§G.1.5).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Download } from "lucide-react";
import {
  cityScaleLabel,
  formatLogMeta,
  formatPolicyMeta,
  parseLogPath,
  parsePolicyLabel,
  strategyColor,
  type PolicyMeta,
} from "../../utils/simMetadata";
import { barOpacity } from "../../utils/chartHighlight";
import { exportChartPng } from "../../utils/chartExport";
import type { DayLogEntry } from "../../types";

export interface PortfolioEfficiencyRun {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function fmt(n: number, d = 2) {
  return isFinite(n) ? n.toFixed(d) : "—";
}

interface ConfigRow {
  id: string;
  label: string;
  runLabel: string;
  policy: string;
  mean: number;
  std: number;
  policyMeta: PolicyMeta;
  logMetaLabel: string;
}

export function PortfolioEfficiencyRanking({
  runs,
  showErrorBars = false,
  logScale = false,
  brushed,
  onPolicyClick,
  onConfigClick,
  maxRows = 24,
}: {
  runs: PortfolioEfficiencyRun[];
  showErrorBars?: boolean;
  logScale?: boolean;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  /** Brush both policy and ``run_label`` for portfolio SQL sync (§G.6). */
  onConfigClick?: (policy: string, runLabel: string) => void;
  maxRows?: number;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const ranked = useMemo(() => {
    const rows: ConfigRow[] = [];
    for (const run of runs) {
      const logMeta = parseLogPath(run.path);
      const graph = cityScaleLabel(logMeta);
      const policies = [...new Set(run.entries.map((e) => e.policy))];
      for (const policy of policies) {
        const vals = run.entries
          .filter((e) => e.policy === policy)
          .map((e) => e.data["kg/km"])
          .filter((v): v is number => v != null);
        if (!vals.length) continue;
        const policyMeta = parsePolicyLabel(policy);
        rows.push({
          id: `${run.path}::${policy}`,
          label: `${graph} · ${policyMeta.selectionStrategy}`,
          runLabel: run.label,
          policy,
          mean: mean(vals),
          std: std(vals),
          policyMeta,
          logMetaLabel: formatLogMeta(logMeta),
        });
      }
    }
    return rows.sort((a, b) => b.mean - a.mean).slice(0, maxRows);
  }, [runs, maxRows]);

  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      grid: { left: 120, right: 24, top: 12, bottom: 12 },
      xAxis: {
        type: (logScale ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: "kg/km",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      yAxis: {
        type: "category" as const,
        data: ranked.map((r) => r.label),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 8 },
      },
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ dataIndex: number }>)[0];
          const r = ranked[p.dataIndex];
          return [
            r.policy,
            r.logMetaLabel,
            formatPolicyMeta(r.policyMeta),
            `${fmt(r.mean)} ± ${fmt(r.std)} kg/km`,
          ].join("<br/>");
        },
      },
      series: [
        {
          type: "bar" as const,
          data: ranked.map((r) => ({
            value: logScale ? Math.max(r.mean, 0.001) : r.mean,
            name: r.policy,
            itemStyle: {
              color: strategyColor(r.policy, { [r.policy]: r.policyMeta }),
              opacity: barOpacity(r.policy, brushed ?? null),
            },
          })),
        },
        ...(showErrorBars && !logScale
          ? [
              {
                type: "custom" as const,
                renderItem: (
                  params: { dataIndex: number },
                  api: {
                    coord: (v: [number, number]) => [number, number];
                    style: (s: object) => object;
                  }
                ) => {
                  const i = params.dataIndex;
                  const r = ranked[i];
                  const y = api.coord([r.mean, i])[1];
                  const xLeft = api.coord([Math.max(0, r.mean - r.std), i])[0];
                  const xRight = api.coord([r.mean + r.std, i])[0];
                  const cap = 4;
                  return {
                    type: "group",
                    children: [
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y, x2: xRight, y2: y },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y - cap, x2: xLeft, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xRight, y1: y - cap, x2: xRight, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                    ],
                  };
                },
              },
            ]
          : []),
      ],
    }),
    [ranked, showErrorBars, logScale, brushed]
  );

  if (ranked.length === 0) return null;

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-gray-300">Portfolio Efficiency Ranking (§G.1.5)</p>
          <p className="text-[10px] text-canvas-muted">
            Top {ranked.length} run×policy configs by mean kg/km
          </p>
        </div>
        <button
          onClick={() =>
            exportChartPng({ current: chartRef.current }, "summary-portfolio-efficiency.png")
          }
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.min(480, 80 + ranked.length * 22) }}
        onEvents={
          onPolicyClick || onConfigClick
            ? {
                click: (params: { dataIndex?: number; name?: string }) => {
                  const row =
                    params.dataIndex != null ? ranked[params.dataIndex] : undefined;
                  if (row && onConfigClick) {
                    onConfigClick(row.policy, row.runLabel);
                    return;
                  }
                  if (params.name && onPolicyClick) onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}
