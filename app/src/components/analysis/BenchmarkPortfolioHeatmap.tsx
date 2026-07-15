/**
 * Portfolio-wide policy×metric heatmap aggregating all loaded runs (§G.1.3).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import {
  formatLogMeta,
  formatPolicyMeta,
  parseLogPath,
  parsePolicyLabel,
  type PolicyMeta,
} from "../../utils/simMetadata";
import { isHighlighted } from "../../utils/chartHighlight";
import { ChartExportButtons } from "../common/ChartExportButtons";
import {
  activeHeatmapMetrics,
  buildNormalizedHeatmapCells,
  HEATMAP_VISUAL_MAP,
  type HeatmapMode,
} from "../../utils/heatmapMetrics";
import type { DayLogEntry } from "../../types";

export interface PortfolioHeatmapRun {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function fmt(n: number, d = 2) {
  return isFinite(n) ? n.toFixed(d) : "—";
}

const MODE_OPTS: Array<{ key: HeatmapMode; label: string }> = [
  { key: "all", label: "All metrics" },
  { key: "overflows", label: "Overflows" },
  { key: "kg/km", label: "kg/km" },
];

export function BenchmarkPortfolioHeatmap({
  runs,
  heatmapMode,
  onModeChange,
  brushed,
  logScale = false,
}: {
  runs: PortfolioHeatmapRun[];
  heatmapMode: HeatmapMode;
  onModeChange?: (mode: HeatmapMode) => void;
  brushed?: string[] | null;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const policies = useMemo(
    () => [...new Set(runs.flatMap((r) => r.entries.map((e) => e.policy)))],
    [runs]
  );

  const policyMeta = useMemo(() => {
    const map: Record<string, PolicyMeta> = {};
    for (const p of policies) map[p] = parsePolicyLabel(p);
    return map;
  }, [policies]);

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
      (p) => (isHighlighted(p, brushed ?? null) ? 1 : 0.15),
      logScale
    );

    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 24, top: 8, bottom: 48 },
      xAxis: {
        type: "category",
        data: policies,
        axisLabel: { color: "#9090b0", fontSize: 8, rotate: 30 },
      },
      yAxis: {
        type: "category",
        data: metrics.map((m) => m.label),
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: HEATMAP_VISUAL_MAP,
      series: [
        {
          type: "heatmap",
          data: cells,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 6, shadowColor: "rgba(0,0,0,0.4)" } },
        },
      ],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const [pi, mi] = p.value;
          const policy = policies[pi];
          const meta = policyMeta[policy];
          const runCount = runs.filter((r) => r.entries.some((e) => e.policy === policy)).length;
          return [
            policy,
            meta ? formatPolicyMeta(meta) : "",
            `${metrics[mi].label}: ${fmt(raw[mi][pi], 2)}`,
            `${runCount} run(s)`,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [policies, runs, metrics, brushed, policyMeta, logScale]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="text-xs font-semibold text-gray-300">Portfolio Policy×Metric Heatmap (§G.1.3)</p>
          <p className="text-[10px] text-canvas-muted">
            {runs.length} runs · {policies.length} policies
            {logScale ? " · log-normalised cells" : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {onModeChange && (
            <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
              {MODE_OPTS.map((o) => (
                <button
                  key={o.key}
                  onClick={() => onModeChange(o.key)}
                  className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                    heatmapMode === o.key
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
          )}
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem="portfolio-heatmap"
          />
        </div>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(140, metrics.length * 36 + 60) }}
      />
      <p className="text-[10px] text-canvas-muted font-mono truncate">
        {runs.map((r) => formatLogMeta(parseLogPath(r.path))).join(" · ")}
      </p>
    </div>
  );
}
