/**
 * Simulation Summary — post-run aggregate analytics (§G.1 / §G.16).
 *
 * Displays:
 *  • Per-policy mean ± std KPI cards
 *  • Policy ranking table (sortable by any metric)
 *  • Per-day trajectory overlay chart (overflows / profit over simulation days)
 *  • Four bar charts: profit, km, overflows, kg/km
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, ChevronUp, ChevronDown, Download } from "lucide-react";
import { useAppStore } from "../../store/app";
import { recentFileLabel, useRecentFilesStore } from "../../store/recentFiles";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { exportChartPng } from "../../utils/chartExport";
import { paretoFront, paretoStepLine } from "../../utils/pareto";
import { symlog } from "../../utils/symlog";
import { barOpacity, isHighlighted, toggleBrush } from "../../utils/chartHighlight";
import {
  citySymbol,
  formatLogMeta,
  formatPolicyMeta,
  parseLogPath,
  parsePolicyLabel,
  strategyColor,
  type LogPathMeta,
  type PolicyMeta,
} from "../../utils/simMetadata";
import {
  buildPolicyHierarchy,
  childrenAtPath,
  enrichDrillChildren,
  policiesAtPath,
  type HierarchyColorMode,
} from "../../utils/policyHierarchy";
import { downloadCsv, downloadParquetTable } from "../../utils/tableExport";
import { buildPolicyParallelAxes } from "../../utils/parallelPolicyAxes";
import { toast } from "sonner";
import type { DayLogEntry } from "../../types";

type HierarchyView = "sunburst" | "treemap";

// ── Stat helpers ──────────────────────────────────────────────────────────────

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function fmt(n: number, decimals = 2): string {
  return isFinite(n) ? n.toFixed(decimals) : "—";
}

// ── Aggregation helpers ───────────────────────────────────────────────────────

interface PolicyStats {
  profit: number[];
  km: number[];
  overflows: number[];
  kg: number[];
  "kg/km": number[];
  cost: number[];
  days: number;
}

type MetricKey = "profit" | "km" | "overflows" | "kg";

function aggregateByPolicy(entries: DayLogEntry[]): Record<string, PolicyStats> {
  const map: Record<string, PolicyStats> = {};
  for (const e of entries) {
    if (!map[e.policy]) {
      map[e.policy] = { profit: [], km: [], overflows: [], kg: [], "kg/km": [], cost: [], days: 0 };
    }
    const d = e.data;
    if (d.profit != null) map[e.policy].profit.push(d.profit);
    if (d.km != null) map[e.policy].km.push(d.km);
    if (d.overflows != null) map[e.policy].overflows.push(d.overflows);
    if (d.kg != null) map[e.policy].kg.push(d.kg);
    if (d["kg/km"] != null) map[e.policy]["kg/km"].push(d["kg/km"]);
    if (d.cost != null) map[e.policy].cost.push(d.cost);
    map[e.policy].days++;
  }
  return map;
}

/** Group entries by (policy, day) and average across samples for trajectory charts. */
function trajectoryByPolicyAndDay(
  entries: DayLogEntry[],
  metric: keyof DayLogEntry["data"]
): Record<string, Array<[number, number]>> {
  const acc: Record<string, Record<number, number[]>> = {};
  for (const e of entries) {
    const val = e.data[metric] as number | undefined;
    if (val == null) continue;
    if (!acc[e.policy]) acc[e.policy] = {};
    if (!acc[e.policy][e.day]) acc[e.policy][e.day] = [];
    acc[e.policy][e.day].push(val);
  }
  const result: Record<string, Array<[number, number]>> = {};
  for (const policy of Object.keys(acc)) {
    result[policy] = Object.entries(acc[policy])
      .map(([day, vals]) => [Number(day), mean(vals)] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  }
  return result;
}

// ── Palette ───────────────────────────────────────────────────────────────────

const POLICY_COLORS = [
  "#6366f1", "#34d399", "#f87171", "#fbbf24",
  "#a78bfa", "#fb923c", "#38bdf8", "#f472b6",
];

function policyTooltipFooter(
  policy: string,
  policyMeta: Record<string, PolicyMeta> | undefined,
  days: number
): string {
  const meta = policyMeta?.[policy];
  const lines = meta ? [formatPolicyMeta(meta)] : [];
  lines.push(`Days: ${days}`);
  return lines.join("<br/>");
}

function ConfigMetaBanner({ logPath, logMeta }: { logPath: string; logMeta: LogPathMeta }) {
  return (
    <div className="card flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-canvas-muted">
      <span className="font-semibold text-gray-300">Run config</span>
      <span>{formatLogMeta(logMeta)}</span>
      <span className="font-mono truncate opacity-70">{logPath.split("/").slice(-4).join("/")}</span>
    </div>
  );
}

function PolicyBrushBar({
  policies,
  brushed,
  onToggle,
  onClear,
}: {
  policies: string[];
  brushed: string[] | null;
  onToggle: (policy: string) => void;
  onClear: () => void;
}) {
  const active = brushed ?? [];
  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Cross-filter (§G.1)</p>
        {active.length > 0 && (
          <button onClick={onClear} className="btn-ghost text-xs">
            Clear filter
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {policies.map((p, i) => {
          const on = active.length === 0 || active.includes(p);
          return (
            <button
              key={p}
              onClick={() => onToggle(p)}
              className={`text-xs px-2 py-0.5 rounded-full border transition-opacity ${
                on ? "opacity-100" : "opacity-35"
              }`}
              style={{
                borderColor: POLICY_COLORS[i % POLICY_COLORS.length],
                color: on ? POLICY_COLORS[i % POLICY_COLORS.length] : undefined,
              }}
            >
              {parsePolicyLabel(p).selectionStrategy}
            </button>
          );
        })}
      </div>
      <p className="text-[10px] text-canvas-muted">
        Click a policy chip or any bar chart to cross-filter all panels.
      </p>
    </div>
  );
}

interface GroupRow {
  label: string;
  mean: number;
  std: number;
  policies: string[];
}

function buildGroupedStats(
  policies: string[],
  stats: Record<string, PolicyStats>,
  groupFn: (p: string) => string,
  metric: "overflows" | "kg/km"
): GroupRow[] {
  const map = new Map<string, string[]>();
  for (const p of policies) {
    const g = groupFn(p);
    const list = map.get(g) ?? [];
    list.push(p);
    map.set(g, list);
  }
  return [...map.entries()].map(([label, ps]) => {
    const vals = ps.flatMap((p) => stats[p][metric]);
    return { label, policies: ps, mean: mean(vals), std: std(vals) };
  });
}

function GroupedMetricBarChart({
  title,
  subtitle,
  groups,
  color,
  showErrorBars,
  exportName,
}: {
  title: string;
  subtitle?: string;
  groups: GroupRow[];
  color: string;
  showErrorBars?: boolean;
  exportName: string;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const labels = groups.map((g) => g.label);

  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ dataIndex: number }>)[0];
          const g = groups[p.dataIndex];
          return `${g.label}<br/>${fmt(g.mean, 2)} ± ${fmt(g.std, 2)}<br/>${g.policies.length} policy variant(s)`;
        },
      },
      grid: { left: 50, right: 10, top: 24, bottom: 40 },
      xAxis: {
        type: "category" as const,
        data: labels,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: "value" as const,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      series: [
        {
          type: "bar" as const,
          data: groups.map((g) => g.mean),
          itemStyle: { color },
        },
        ...(showErrorBars
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
                  const g = groups[params.dataIndex];
                  const x = api.coord([params.dataIndex, g.mean])[0];
                  const yTop = api.coord([params.dataIndex, g.mean + g.std])[1];
                  const yBot = api.coord([params.dataIndex, Math.max(0, g.mean - g.std)])[1];
                  const cap = 5;
                  return {
                    type: "group",
                    children: [
                      {
                        type: "line",
                        shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                    ],
                  };
                },
                data: groups.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [groups, labels, color, showErrorBars]
  );

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <div>
          <p className="text-xs text-canvas-muted">{title}</p>
          {subtitle && <p className="text-[10px] text-canvas-muted">{subtitle}</p>}
        </div>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, `${exportName}.png`)}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

type SortKey = MetricKey;
type SortDir = "asc" | "desc";

function RankingTable({
  stats,
  policies,
  onExport,
  onExportParquet,
  parquetExporting,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  onExport: () => void;
  onExportParquet?: () => void;
  parquetExporting?: boolean;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("profit");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    return [...policies].sort((a, b) => {
      const va = mean(stats[a][sortKey]);
      const vb = mean(stats[b][sortKey]);
      return sortDir === "desc" ? vb - va : va - vb;
    });
  }, [policies, stats, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  function setDir(fn: (d: SortDir) => SortDir) {
    setSortDir(fn);
  }

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (k !== sortKey) return <span className="w-3" />;
    return sortDir === "desc" ? <ChevronDown size={11} /> : <ChevronUp size={11} />;
  };

  const cols: Array<{ key: SortKey; label: string; unit: string }> = [
    { key: "profit", label: "Profit", unit: "€" },
    { key: "km", label: "Distance", unit: "km" },
    { key: "overflows", label: "Overflows", unit: "" },
    { key: "kg", label: "Waste", unit: "kg" },
  ];

  return (
    <div className="card overflow-x-auto">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-gray-300">Policy Ranking</p>
        <div className="flex items-center gap-2">
          <button onClick={onExport} className="btn-ghost text-xs flex items-center gap-1">
            <Download size={12} />
            Export CSV
          </button>
          {onExportParquet && (
            <button
              onClick={onExportParquet}
              disabled={parquetExporting}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              Export Parquet
            </button>
          )}
        </div>
      </div>
      <table className="w-full text-xs min-w-[520px]">
        <thead>
          <tr className="border-b border-canvas-border text-left">
            <th className="py-2 px-3 text-canvas-muted font-medium w-8">#</th>
            <th className="py-2 px-3 text-canvas-muted font-medium">Policy</th>
            {cols.map((c) => (
              <th
                key={c.key}
                className="py-2 px-3 text-canvas-muted font-medium cursor-pointer hover:text-gray-300 select-none"
                onClick={() => toggleSort(c.key)}
              >
                <span className="flex items-center gap-1">
                  Mean {c.label} {c.unit ? `(${c.unit})` : ""}
                  <SortIcon k={c.key} />
                </span>
              </th>
            ))}
            <th className="py-2 px-3 text-canvas-muted font-medium">Days</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-canvas-border">
          {sorted.map((policy, rank) => {
            const s = stats[policy];
            const color = POLICY_COLORS[policies.indexOf(policy) % POLICY_COLORS.length];
            return (
              <tr key={policy} className="hover:bg-canvas-hover">
                <td className="py-1.5 px-3 text-canvas-muted">{rank + 1}</td>
                <td className="py-1.5 px-3">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                    <span className="font-mono text-accent-secondary">{policy}</span>
                  </span>
                </td>
                {cols.map((c) => {
                  const vals = s[c.key];
                  const m = mean(vals);
                  const sd = std(vals);
                  return (
                    <td key={c.key} className="py-1.5 px-3 font-mono text-gray-300">
                      {fmt(m, 1)}
                      {sd > 0 && (
                        <span className="text-canvas-muted text-[10px] ml-1">±{fmt(sd, 1)}</span>
                      )}
                    </td>
                  );
                })}
                <td className="py-1.5 px-3 text-canvas-muted">{s.days}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

type TrajectoryMetric = "overflows" | "profit" | "km" | "kg";

function TrajectoryChart({
  entries,
  policies,
  brushed,
}: {
  entries: DayLogEntry[];
  policies: string[];
  brushed?: string[] | null;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const [metric, setMetric] = useState<TrajectoryMetric>("overflows");

  const traj = useMemo(
    () => trajectoryByPolicyAndDay(entries, metric),
    [entries, metric]
  );

  const allDays = useMemo(() => {
    const days = new Set<number>();
    for (const pts of Object.values(traj)) pts.forEach(([d]) => days.add(d));
    return [...days].sort((a, b) => a - b);
  }, [traj]);

  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: { data: policies, textStyle: { color: "#9090b0", fontSize: 10 }, bottom: 0 },
    grid: { left: 50, right: 10, top: 20, bottom: 60 },
    xAxis: {
      type: "category" as const,
      data: allDays,
      name: "Day",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 9 },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    series: policies.map((policy, i) => ({
      name: policy,
      type: "line" as const,
      smooth: true,
      symbol: "none",
      lineStyle: { width: 1.5, opacity: barOpacity(policy, brushed ?? null) },
      color: POLICY_COLORS[i % POLICY_COLORS.length],
      data: allDays.map((day) => {
        const pt = traj[policy]?.find(([d]) => d === day);
        return pt ? pt[1] : null;
      }),
    })),
  }), [traj, allDays, policies, brushed]);

  const METRIC_OPTS: Array<{ key: TrajectoryMetric; label: string }> = [
    { key: "overflows", label: "Overflows" },
    { key: "profit", label: "Profit (€)" },
    { key: "km", label: "Distance (km)" },
    { key: "kg", label: "Waste (kg)" },
  ];

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs font-semibold text-gray-300">Per-Day Trajectory</p>
        <div className="flex items-center gap-2">
          {allDays.length > 0 && (
            <button
              onClick={() => exportChartPng({ current: chartRef.current }, `trajectory-${metric}.png`)}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              PNG
            </button>
          )}
        <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
          {METRIC_OPTS.map((o) => (
            <button
              key={o.key}
              onClick={() => setMetric(o.key)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                metric === o.key
                  ? "bg-accent-primary text-white"
                  : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
        </div>
      </div>
      {allDays.length === 0 ? (
        <p className="text-xs text-canvas-muted">No day data available.</p>
      ) : (
        <ReactECharts ref={chartRef} option={option} style={{ height: 260 }} />
      )}
    </div>
  );
}

const RADAR_METRICS: Array<{ key: MetricKey; label: string }> = [
  { key: "profit", label: "Profit (€)" },
  { key: "km", label: "Distance (km)" },
  { key: "overflows", label: "Overflows" },
  { key: "kg", label: "Waste (kg)" },
];

function PolicyRadarChart({
  stats,
  policies,
  brushed,
  policyMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  brushed?: string[] | null;
  policyMeta?: Record<string, PolicyMeta>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const option = useMemo(() => {
    const metricMeans: Record<string, Record<string, number>> = {};
    for (const p of policies) {
      metricMeans[p] = {};
      for (const { key } of RADAR_METRICS) {
        metricMeans[p][key] = mean(stats[p][key]);
      }
    }
    const maxes = RADAR_METRICS.map(({ key }) =>
      Math.max(...policies.map((p) => metricMeans[p][key] ?? 0), 1)
    );

    return {
      backgroundColor: "transparent",
      legend: { data: policies, textStyle: { color: "#9090b0", fontSize: 10 } },
      radar: {
        indicator: RADAR_METRICS.map(({ label }, i) => ({ name: label, max: maxes[i] * 1.1 })),
        axisLine: { lineStyle: { color: "#2d2d50" } },
        splitLine: { lineStyle: { color: "#2d2d50" } },
        name: { textStyle: { color: "#9090b0", fontSize: 9 } },
      },
      series: [
        {
          type: "radar" as const,
          data: policies.map((p, i) => ({
            name: p,
            value: RADAR_METRICS.map(({ key }) => metricMeans[p][key] ?? 0),
            lineStyle: {
              color: POLICY_COLORS[i % POLICY_COLORS.length],
              opacity: barOpacity(p, brushed ?? null),
            },
            areaStyle: {
              color: `${POLICY_COLORS[i % POLICY_COLORS.length]}20`,
              opacity: barOpacity(p, brushed ?? null),
            },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const meta = policyMeta?.[p.name];
          return meta
            ? `${p.name}<br/>${formatPolicyMeta(meta)}`
            : p.name;
        },
      },
    };
  }, [stats, policies, brushed, policyMeta]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Policy Radar</p>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, "summary-radar.png")}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 280 }} />
    </div>
  );
}

const HEATMAP_METRICS: Array<{ key: MetricKey | "kg/km"; label: string; higherBetter: boolean }> = [
  { key: "profit", label: "Profit", higherBetter: true },
  { key: "kg/km", label: "kg/km", higherBetter: true },
  { key: "overflows", label: "Overflows", higherBetter: false },
  { key: "km", label: "km", higherBetter: false },
];

type HeatmapMode = "all" | "overflows" | "kg/km";

function PolicyHeatmapChart({
  stats,
  policies,
  mode,
  onModeChange,
  brushed,
  policyMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  mode: HeatmapMode;
  onModeChange?: (mode: HeatmapMode) => void;
  brushed?: string[] | null;
  policyMeta?: Record<string, PolicyMeta>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const activeMetrics = useMemo(
    () =>
      mode === "all"
        ? HEATMAP_METRICS
        : HEATMAP_METRICS.filter((m) => m.key === mode),
    [mode]
  );

  const option = useMemo(() => {
    const cells: Array<[number, number, number]> = [];
    for (let mi = 0; mi < activeMetrics.length; mi++) {
      const { key, higherBetter } = activeMetrics[mi];
      const raw = policies.map((p) =>
        key === "kg/km" ? mean(stats[p]["kg/km"]) : mean(stats[p][key as MetricKey])
      );
      const min = Math.min(...raw);
      const max = Math.max(...raw);
      const span = max - min || 1;
      for (let pi = 0; pi < policies.length; pi++) {
        let norm = (raw[pi] - min) / span;
        if (!higherBetter) norm = 1 - norm;
        if (!isHighlighted(policies[pi], brushed ?? null)) norm *= 0.15;
        cells.push([pi, mi, norm]);
      }
    }

    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 24, top: 8, bottom: 48 },
      xAxis: {
        type: "category" as const,
        data: policies,
        axisLabel: { color: "#9090b0", fontSize: 8, rotate: 30 },
      },
      yAxis: {
        type: "category" as const,
        data: activeMetrics.map((m) => m.label),
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: false,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
        textStyle: { color: "#9090b0", fontSize: 9 },
        show: false,
      },
      series: [
        {
          type: "heatmap" as const,
          data: cells,
          label: { show: false },
          emphasis: {
            itemStyle: { shadowBlur: 6, shadowColor: "rgba(0,0,0,0.4)" },
          },
        },
      ],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const [pi, mi, norm] = p.value;
          const { key, label } = activeMetrics[mi];
          const policy = policies[pi];
          const raw =
            key === "kg/km"
              ? mean(stats[policy]["kg/km"])
              : mean(stats[policy][key as MetricKey]);
          const meta = policyMeta?.[policy];
          return [
            policy,
            meta ? formatPolicyMeta(meta) : "",
            `${label}: ${fmt(raw, 2)}`,
            `Score: ${fmt(norm * 100, 0)}%`,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [stats, policies, activeMetrics, brushed, policyMeta]);

  const MODE_OPTS: Array<{ key: HeatmapMode; label: string }> = [
    { key: "all", label: "All metrics" },
    { key: "overflows", label: "Overflows" },
    { key: "kg/km", label: "kg/km" },
  ];

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs font-semibold text-gray-300">Policy Metric Heatmap</p>
        <div className="flex items-center gap-2">
          {onModeChange && (
            <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
              {MODE_OPTS.map((o) => (
                <button
                  key={o.key}
                  onClick={() => onModeChange(o.key)}
                  className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                    mode === o.key
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
          )}
          <button
            onClick={() => exportChartPng({ current: chartRef.current }, "summary-heatmap.png")}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Download size={12} />
            PNG
          </button>
        </div>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(120, activeMetrics.length * 36 + 60) }}
      />
    </div>
  );
}

function PolicyParallelChart({
  stats,
  policies,
  policyMeta,
  logMeta,
  brushed,
  onPolicyClick,
  onAxisBrush,
  onOverflowCorridorBrush,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  onAxisBrush?: (policies: string[] | null) => void;
  /** Sync zero-overflow corridor slider when brushing the overflows axis (§G.1.4). */
  onOverflowCorridorBrush?: (maxOverflow: number | null) => void;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const parallel = useMemo(
    () => buildPolicyParallelAxes(policies, stats, policyMeta, logMeta),
    [policies, stats, policyMeta, logMeta]
  );

  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      brush: {
        toolbox: ["parallelAxis", "clear"],
        parallelAxisIndex: "all",
        throttleType: "debounce",
        throttleDelay: 300,
      },
      toolbox: {
        right: 8,
        top: 0,
        itemSize: 12,
        iconStyle: { borderColor: "#9090b0" },
        feature: {
          brush: { type: ["parallelAxis", "clear"] },
        },
      },
      parallelAxis: parallel.axes.map((s) => ({
        dim: s.dim,
        name: s.name,
        type: s.type,
        ...(s.data ? { data: s.data } : {}),
        ...(s.max != null ? { max: s.max } : {}),
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLine: { lineStyle: { color: "#2d2d50" } },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      })),
      parallel: {
        left: 50,
        right: 24,
        top: 28,
        bottom: 36,
      },
      series: [
        {
          type: "parallel" as const,
          lineStyle: { width: 2 },
          data: parallel.rows.map((row) => ({
            name: row.name,
            value: row.value,
            lineStyle: {
              color: strategyColor(row.name, policyMeta),
              opacity: barOpacity(row.name, brushed ?? null),
            },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const meta = policyMeta[p.name];
          return meta ? `${p.name}<br/>${formatPolicyMeta(meta)}` : p.name;
        },
      },
    };
  }, [parallel, policyMeta, brushed]);

  const events = useMemo(() => {
    const handlers: Record<string, (params: unknown) => void> = {};
    if (onPolicyClick) {
      handlers.click = (params: unknown) => {
        const p = params as { name?: string };
        if (p.name) onPolicyClick(p.name);
      };
    }
    if (onAxisBrush) {
      handlers.brushselected = (params: unknown) => {
        const batch = (params as { batch?: Array<{ selected?: Array<{ dataIndex?: number[] }> }> })
          .batch;
        const indices = batch?.[0]?.selected?.[0]?.dataIndex;
        if (!indices?.length) {
          onAxisBrush(null);
          return;
        }
        onAxisBrush(indices.map((i) => policies[i]).filter(Boolean));
      };
    }
    if (onOverflowCorridorBrush) {
      handlers.brushEnd = (params: unknown) => {
        const areas = (params as { areas?: Array<{ parallelAxisIndex?: number; coordRange?: number[] }> })
          .areas;
        if (!areas?.length) {
          onOverflowCorridorBrush(null);
          return;
        }
        for (const area of areas) {
          if (area.parallelAxisIndex !== parallel.overflowDim || !area.coordRange?.length) continue;
          const range = area.coordRange;
          const hi = Math.max(range[0], range[range.length - 1]);
          onOverflowCorridorBrush(hi);
          return;
        }
      };
    }
    return Object.keys(handlers).length ? handlers : undefined;
  }, [onPolicyClick, onAxisBrush, onOverflowCorridorBrush, policies, parallel.overflowDim]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-gray-300">Policy Parallel Coordinates</p>
          {(onAxisBrush || onOverflowCorridorBrush) && (
            <p className="text-[10px] text-canvas-muted">
              Drag on any axis to brush · overflows axis syncs corridor slider · toolbox clear resets
            </p>
          )}
        </div>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, "summary-parallel.png")}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 220 }}
        onEvents={events}
      />
    </div>
  );
}

function HierarchyBreadcrumb({
  path,
  onNavigate,
}: {
  path: string[];
  onNavigate: (depth: number) => void;
}) {
  if (path.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      {path.map((seg, i) => (
        <span key={`${seg}-${i}`} className="flex items-center gap-1">
          {i > 0 && <span className="text-canvas-muted">›</span>}
          <button
            onClick={() => onNavigate(i + 1)}
            className="text-accent-secondary hover:underline font-mono"
          >
            {seg}
          </button>
        </span>
      ))}
    </div>
  );
}

function PolicyHierarchyPanel({
  stats,
  policies,
  policyMeta,
  logMeta,
  brushed,
  onBrushPolicies,
  showErrorBars = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  brushed?: string[] | null;
  onBrushPolicies: (ps: string[]) => void;
  showErrorBars?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const drillRef = useRef<EChartsReact | null>(null);
  const [view, setView] = useState<HierarchyView>("sunburst");
  const [colorMode, setColorMode] = useState<HierarchyColorMode>("kgkm");
  const [drillPath, setDrillPath] = useState<string[]>([]);

  const tree = useMemo(
    () => buildPolicyHierarchy(policies, stats, policyMeta, logMeta, colorMode),
    [policies, stats, policyMeta, logMeta, colorMode]
  );

  const hierarchyOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: {
        formatter: (p: { name: string; value: number }) =>
          `${p.name}<br/>Profit: ${fmt(p.value, 1)} €`,
      },
      series: [
        view === "sunburst"
          ? {
              type: "sunburst" as const,
              data: tree,
              radius: ["12%", "90%"],
              label: { color: "#c0c0d8", fontSize: 9 },
              itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
              emphasis: { focus: "ancestor" as const },
            }
          : {
              type: "treemap" as const,
              data: tree,
              leafDepth: 2,
              label: { color: "#c0c0d8", fontSize: 9 },
              upperLabel: { show: true, height: 22, color: "#9090b0" },
              itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
            },
      ],
    }),
    [tree, view]
  );

  const drillChildren = useMemo(
    () => enrichDrillChildren(childrenAtPath(tree, drillPath), stats, policyMeta),
    [tree, drillPath, stats, policyMeta]
  );

  const drillOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      grid: { left: 110, right: 24, top: 12, bottom: 12 },
      xAxis: {
        type: "value" as const,
        name: "Profit (€)",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: "category" as const,
        data: drillChildren.map((c) => c.name),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ dataIndex: number }>)[0];
          const c = drillChildren[p.dataIndex];
          const err = Math.max(c.profitStd, c.distSpread);
          return [
            `${c.name}`,
            `Profit: ${fmt(c.profit, 1)} € ± ${fmt(err, 1)}`,
            `kg/km: ${fmt(c.kgkm, 2)}`,
            `Overflows: ${fmt(c.overflows, 1)}`,
            c.distSpread > 0 ? `Empirical↔Gamma spread: ${fmt(c.distSpread, 1)} €` : "",
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
      series: [
        {
          type: "bar" as const,
          data: drillChildren.map((c) => ({
            value: c.profit,
            name: c.name,
            itemStyle: {
              opacity: c.policies.some((p) => isHighlighted(p, brushed ?? null)) ? 1 : 0.25,
            },
          })),
          itemStyle: { color: "#6366f1" },
        },
        ...(showErrorBars
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
                  const c = drillChildren[params.dataIndex];
                  const err = Math.max(c.profitStd, c.distSpread);
                  const y = api.coord([c.profit, params.dataIndex])[1];
                  const xLeft = api.coord([Math.max(0, c.profit - err), params.dataIndex])[0];
                  const xRight = api.coord([c.profit + err, params.dataIndex])[0];
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
                data: drillChildren.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [drillChildren, brushed, showErrorBars]
  );

  const handleSegmentClick = (path: string[]) => {
    setDrillPath(path);
    onBrushPolicies(policiesAtPath(tree, path));
  };

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="text-xs font-semibold text-gray-300">Policy Hierarchy (§G.2)</p>
          <p className="text-[10px] text-canvas-muted">
            Span = profit · color ={" "}
            {colorMode === "kgkm" ? "kg/km efficiency" : "mean overflows"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["kgkm", "overflows"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setColorMode(m)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  colorMode === m
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {m === "kgkm" ? "kg/km" : "Overflows"}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["sunburst", "treemap"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`text-xs px-2.5 py-1 rounded-md capitalize transition-colors ${
                  view === v
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          <button
            onClick={() =>
              exportChartPng({ current: chartRef.current }, `summary-${view}.png`)
            }
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Download size={12} />
            PNG
          </button>
        </div>
      </div>

      <HierarchyBreadcrumb
        path={drillPath}
        onNavigate={(depth) => {
          const next = drillPath.slice(0, depth);
          setDrillPath(next);
          onBrushPolicies(next.length ? policiesAtPath(tree, next) : policies);
        }}
      />

      <ReactECharts
        ref={chartRef}
        option={hierarchyOption}
        style={{ height: 300 }}
        onEvents={{
          click: (params: { treePathInfo?: Array<{ name: string }> }) => {
            const path = params.treePathInfo?.map((n) => n.name) ?? [];
            if (path.length) handleSegmentClick(path);
          },
        }}
      />

      {drillPath.length > 0 && drillChildren.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-canvas-muted">Drill-down variants</p>
          <ReactECharts
            ref={drillRef}
            option={drillOption}
            style={{ height: Math.max(120, drillChildren.length * 28) }}
            onEvents={{
              click: (params: { name?: string }) => {
                if (!params.name) return;
                const child = drillChildren.find((c) => c.name === params.name);
                if (child) onBrushPolicies(child.policies);
              },
            }}
          />
        </div>
      )}
    </div>
  );
}

function DistributionFacetHeatmaps({
  stats,
  policies,
  policyMeta,
  heatmapMode,
  onModeChange,
  brushed,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  heatmapMode: HeatmapMode;
  onModeChange: (mode: HeatmapMode) => void;
  brushed?: string[] | null;
}) {
  const facets = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const p of policies) {
      const dist = policyMeta[p]?.distribution ?? "—";
      const list = map.get(dist) ?? [];
      list.push(p);
      map.set(dist, list);
    }
    return [...map.entries()];
  }, [policies, policyMeta]);

  if (facets.length <= 1) {
    return (
      <PolicyHeatmapChart
        stats={stats}
        policies={policies}
        mode={heatmapMode}
        onModeChange={onModeChange}
        brushed={brushed}
        policyMeta={policyMeta}
      />
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-gray-300">Policy Heatmaps by Distribution (§G.1.3)</p>
      <div className={`grid gap-4 ${facets.length >= 2 ? "grid-cols-1 lg:grid-cols-2" : ""}`}>
        {facets.map(([dist, ps]) => (
          <div key={dist} className="space-y-1">
            <p className="text-[10px] text-canvas-muted font-mono">{dist}</p>
            <PolicyHeatmapChart
              stats={stats}
              policies={ps}
              mode={heatmapMode}
              onModeChange={onModeChange}
              brushed={brushed}
              policyMeta={policyMeta}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function EfficiencyRankingChart({
  stats,
  policies,
  showErrorBars = false,
  brushed,
  onPolicyClick,
  policyMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  showErrorBars?: boolean;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const ranked = useMemo(
    () =>
      [...policies]
        .map((p) => ({
          policy: p,
          mean: mean(stats[p]["kg/km"]),
          std: std(stats[p]["kg/km"]),
        }))
        .sort((a, b) => b.mean - a.mean),
    [stats, policies]
  );

  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      grid: { left: 110, right: 24, top: 12, bottom: 12 },
      xAxis: {
        type: "value" as const,
        name: "kg/km",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: "category" as const,
        data: ranked.map((r) => r.policy),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ name: string; value: number; dataIndex: number }>)[0];
          const r = ranked[p.dataIndex];
          const days = stats[r.policy]?.days ?? 0;
          const footer = policyTooltipFooter(r.policy, policyMeta, days);
          return `${r.policy}<br/>${fmt(r.mean, 2)} ± ${fmt(r.std, 2)} kg/km<br/>${footer}`;
        },
      },
      series: [
        {
          type: "bar" as const,
          data: ranked.map((r) => ({
            value: r.mean,
            name: r.policy,
            itemStyle: {
              color: strategyColor(r.policy, policyMeta),
              opacity: barOpacity(r.policy, brushed ?? null),
            },
          })),
        },
        ...(showErrorBars
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
                data: ranked.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [ranked, showErrorBars, brushed, policyMeta, stats]
  );

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Efficiency Ranking (kg/km)</p>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, "summary-efficiency-rank.png")}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(160, ranked.length * 28) }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name) onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

function PolicyParetoChart({
  stats,
  policies,
  logScale = false,
  brushed,
  onPolicyClick,
  policyMeta,
  logMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  logScale?: boolean;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
  logMeta?: LogPathMeta | null;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const option = useMemo(() => {
    const points = policies.map((p) => ({
      id: p,
      x: mean(stats[p].profit),
      y: mean(stats[p].overflows),
    }));
    const frontIds = new Set(paretoFront(points).map((p) => p.id));
    const step = paretoStepLine(paretoFront(points));
    const displayY = (y: number) => (logScale ? Math.max(y, 0.001) : y);
    const displayStep = step.map(([x, y]) => [x, displayY(y)] as [number, number]);

    return {
      backgroundColor: "transparent",
      grid: { left: 50, right: 16, top: 24, bottom: 40 },
      xAxis: {
        type: "value" as const,
        name: "Profit (€)",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: (logScale ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: "Overflows",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          name: "Policies",
          type: "scatter" as const,
          data: points.map((pt) => {
            const onFront = frontIds.has(pt.id);
            const highlighted = isHighlighted(pt.id, brushed ?? null);
            return {
              name: pt.id,
              value: [pt.x, displayY(pt.y)],
              itemStyle: {
                color: onFront ? strategyColor(pt.id, policyMeta) : "#6b7280",
                opacity: highlighted ? 1 : 0.2,
              },
              symbol: citySymbol(logMeta ?? null),
              symbolSize: onFront ? 10 : 7,
            };
          }),
          tooltip: {
            formatter: (p: { name: string; value: [number, number] }) => {
              const pt = points.find((x) => x.id === p.name);
              const meta = policyMeta?.[p.name];
              const lines = [
                p.name,
                meta ? formatPolicyMeta(meta) : "",
                `Profit: ${fmt(p.value[0], 1)} €`,
                `Overflows: ${fmt(pt?.y ?? p.value[1], 1)}`,
                frontIds.has(p.name) ? "Pareto-optimal" : "",
              ].filter(Boolean);
              return lines.join("<br/>");
            },
          },
        },
        ...(displayStep.length > 1
          ? [
              {
                name: "Pareto front",
                type: "line" as const,
                data: displayStep,
                lineStyle: { color: "#f3f4f6", type: "dashed" as const, width: 1.5 },
                symbol: "none",
                tooltip: { show: false },
                z: 1,
              },
            ]
          : []),
      ],
      legend: {
        data: displayStep.length > 1 ? ["Policies", "Pareto front"] : ["Policies"],
        textStyle: { color: "#9090b0", fontSize: 9 },
        top: 0,
      },
    };
  }, [stats, policies, logScale, brushed, policyMeta, logMeta]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Profit vs Overflows (Pareto)</p>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, "summary-pareto.png")}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 260 }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name && params.name !== "Pareto front") onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

function MetricBarChart({
  title,
  policies,
  values,
  color,
  logScale = false,
  useSymlog = false,
  showErrorBars = false,
  exportName,
  brushed,
  onPolicyClick,
  policyMeta,
  stats,
}: {
  title: string;
  policies: string[];
  values: Array<{ mean: number; std: number }>;
  color: string;
  logScale?: boolean;
  useSymlog?: boolean;
  showErrorBars?: boolean;
  exportName: string;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
  stats?: Record<string, PolicyStats>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const symlogMode = logScale && useSymlog;
  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ name: string; value: number; dataIndex: number }>)[0];
        const raw = values[p.dataIndex];
        const days = stats?.[p.name]?.days ?? 0;
        const footer = policyTooltipFooter(p.name, policyMeta, days);
        return `${p.name}<br/>${fmt(raw?.mean ?? p.value, 2)} ± ${fmt(raw?.std ?? 0, 2)}<br/>${footer}`;
      },
    },
    xAxis: {
      type: "category" as const,
      data: policies,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 30 },
    },
    yAxis: {
      type: (logScale && !symlogMode ? "log" : "value") as "log" | "value",
      logBase: 10,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    grid: { left: 45, right: 10, top: 16, bottom: 55 },
    series: [
      {
        type: "bar" as const,
        data: policies.map((policy, i) => {
          const v = values[i];
          const display =
            !logScale ? v.mean : symlogMode ? symlog(v.mean) : Math.max(v.mean, 0.001);
          return {
            value: display,
            name: policy,
            itemStyle: { color, opacity: barOpacity(policy, brushed ?? null) },
          };
        }),
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
                const v = values[i];
                const x = api.coord([i, v.mean])[0];
                const yTop = api.coord([i, v.mean + v.std])[1];
                const yBot = api.coord([i, Math.max(0, v.mean - v.std)])[1];
                const cap = 5;
                return {
                  type: "group",
                  children: [
                    {
                      type: "line",
                      shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                    {
                      type: "line",
                      shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                    {
                      type: "line",
                      shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                  ],
                };
              },
              data: policies.map((_, i) => i),
              z: 10,
            },
          ]
        : []),
    ],
  }), [policies, values, color, logScale, symlogMode, showErrorBars, brushed, policyMeta, stats]);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-canvas-muted">{title}</p>
        <button
          onClick={() => exportChartPng({ current: chartRef.current }, `${exportName}.png`)}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <Download size={12} />
          PNG
        </button>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 190 }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name) onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function SimulationSummary() {
  const { pendingLogPath, setPendingLogPath, projectRoot } = useAppStore();
  const [parquetExporting, setParquetExporting] = useState(false);
  const [logScale, setLogScale] = useState(false);
  const [showErrorBars, setShowErrorBars] = useState(false);
  const [brushed, setBrushed] = useState<string[] | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<HeatmapMode>("all");
  const [overflowMax, setOverflowMax] = useState<number | null>(null);
  const { policy, sampleId } = useGlobalFiltersStore(); // used by filteredEntries
  const [entries, setEntries] = useState<DayLogEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);

  const pushRecent = useRecentFilesStore((s) => s.pushRecent);

  const loadLog = useCallback(async (path: string) => {
    const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    setEntries(loaded);
    setLogPath(path);
    pushRecent({ path, label: recentFileLabel(path), kind: "log" });
  }, [pushRecent]);

  // Auto-load when another page hands off a log path (e.g. OutputBrowser)
  useEffect(() => {
    if (pendingLogPath) {
      loadLog(pendingLogPath);
      setPendingLogPath(null);
    }
  }, [pendingLogPath, setPendingLogPath, loadLog]);

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    loadLog(path);
  }, [loadLog]);

  const filteredEntries = useMemo(
    () => filterEntries(entries, policy, sampleId),
    [entries, policy, sampleId]
  );

  const stats = useMemo(() => aggregateByPolicy(filteredEntries), [filteredEntries]);
  const policies = useMemo(() => Object.keys(stats), [stats]);

  const logMeta = useMemo(() => parseLogPath(logPath), [logPath]);
  const policyMeta = useMemo(() => {
    const map: Record<string, PolicyMeta> = {};
    for (const p of policies) map[p] = parsePolicyLabel(p);
    return map;
  }, [policies]);

  const handlePolicyClick = useCallback((p: string) => {
    setBrushed((prev) => toggleBrush(prev, p));
  }, []);

  const handleBrushPolicies = useCallback(
    (ps: string[] | null) => {
      if (!ps?.length) {
        setBrushed(null);
        return;
      }
      setBrushed(ps.length >= policies.length ? null : ps);
    },
    [policies.length]
  );

  const effectiveBrushed = useMemo(() => {
    if (overflowMax == null) return brushed;
    const corridor = policies.filter((p) => mean(stats[p].overflows) <= overflowMax);
    if (!brushed || brushed.length === 0) return corridor;
    return corridor.filter((p) => brushed.includes(p));
  }, [brushed, overflowMax, policies, stats]);

  const maxOverflow = useMemo(
    () => Math.max(...policies.map((p) => mean(stats[p].overflows)), 0),
    [policies, stats]
  );

  const overflowGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.selectionStrategy ?? "—",
        "overflows"
      ),
    [policies, stats, policyMeta]
  );

  const kgkmGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.constructor ?? "—",
        "kg/km"
      ),
    [policies, stats, policyMeta]
  );

  const rankingExportData = useCallback(() => {
    const cols: MetricKey[] = ["profit", "km", "overflows", "kg"];
    const headers = ["policy", ...cols.map((c) => `mean_${c}`), ...cols.map((c) => `std_${c}`), "days"];
    const rows = policies.map((p) => [
      p,
      ...cols.map((c) => mean(stats[p][c]).toFixed(4)),
      ...cols.map((c) => std(stats[p][c]).toFixed(4)),
      stats[p].days,
    ]);
    return { headers, rows };
  }, [policies, stats]);

  const exportRankingCsv = useCallback(() => {
    const { headers, rows } = rankingExportData();
    downloadCsv("simulation-ranking.csv", headers, rows);
  }, [rankingExportData]);

  const exportRankingParquet = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to export Parquet");
      return;
    }
    const { headers, rows } = rankingExportData();
    setParquetExporting(true);
    try {
      const out = await downloadParquetTable(projectRoot, "simulation-ranking.parquet", headers, rows);
      if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
    } catch (err) {
      toast.error("Parquet export failed", { description: String(err) });
    } finally {
      setParquetExporting(false);
    }
  }, [projectRoot, rankingExportData]);

  const metricValues = (key: MetricKey) =>
    policies.map((p) => ({
      mean: mean(stats[p][key]),
      std: std(stats[p][key]),
    }));

  return (
    <div className="space-y-4">
      <GlobalFilterBar />

      <div className="flex items-center gap-3">
        <button onClick={openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>
        {logPath && (
          <span className="text-xs text-canvas-muted font-mono truncate">{logPath.split("/").pop()}</span>
        )}
      </div>

      {entries.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Load a completed simulation log to view summary analytics.
        </div>
      )}

      {policies.length > 0 && (
        <>
          {logPath && <ConfigMetaBanner logPath={logPath} logMeta={logMeta} />}

          <PolicyBrushBar
            policies={policies}
            brushed={brushed}
            onToggle={handlePolicyClick}
            onClear={() => setBrushed(null)}
          />

          {/* Policy ranking table */}
          <RankingTable
            stats={stats}
            policies={policies}
            onExport={exportRankingCsv}
            onExportParquet={projectRoot ? exportRankingParquet : undefined}
            parquetExporting={parquetExporting}
          />

          <PolicyHierarchyPanel
            stats={stats}
            policies={policies}
            policyMeta={policyMeta}
            logMeta={logMeta}
            brushed={effectiveBrushed}
            onBrushPolicies={handleBrushPolicies}
            showErrorBars={showErrorBars}
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <GroupedMetricBarChart
              title="Overflows by Selection Strategy"
              subtitle="Mean ± std across constructor variants (§G.1.1)"
              groups={overflowGroups}
              color="#f87171"
              showErrorBars={showErrorBars}
              exportName="summary-overflows-grouped"
            />
            <GroupedMetricBarChart
              title="kg/km by Constructor"
              subtitle="Mean ± std across selection strategies (§G.1.1)"
              groups={kgkmGroups}
              color="#34d399"
              showErrorBars={showErrorBars}
              exportName="summary-kgkm-grouped"
            />
          </div>

          {/* Per-day trajectory */}
          <TrajectoryChart entries={filteredEntries} policies={policies} brushed={effectiveBrushed} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <PolicyRadarChart
              stats={stats}
              policies={policies}
              brushed={effectiveBrushed}
              policyMeta={policyMeta}
            />
            <div className="space-y-2">
              <div className="card space-y-2">
                <p className="text-xs font-semibold text-gray-300">Zero-overflow corridor (§G.1.4)</p>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={Math.max(maxOverflow, 1)}
                    step={0.5}
                    value={overflowMax ?? maxOverflow}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setOverflowMax(v >= maxOverflow ? null : v);
                    }}
                    className="flex-1 accent-accent-primary"
                  />
                  <span className="text-xs font-mono text-canvas-muted w-24 text-right">
                    ≤ {overflowMax == null ? "off" : fmt(overflowMax, 1)}
                  </span>
                  {overflowMax != null && (
                    <button onClick={() => setOverflowMax(null)} className="btn-ghost text-xs">
                      Clear
                    </button>
                  )}
                </div>
              </div>
              <PolicyParallelChart
                stats={stats}
                policies={policies}
                policyMeta={policyMeta}
                logMeta={logMeta}
                brushed={effectiveBrushed}
                onPolicyClick={handlePolicyClick}
                onAxisBrush={handleBrushPolicies}
                onOverflowCorridorBrush={(max) => setOverflowMax(max)}
              />
            </div>
          </div>

          <DistributionFacetHeatmaps
            stats={stats}
            policies={policies}
            policyMeta={policyMeta}
            heatmapMode={heatmapMode}
            onModeChange={setHeatmapMode}
            brushed={effectiveBrushed}
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <EfficiencyRankingChart
              stats={stats}
              policies={policies}
              showErrorBars={showErrorBars}
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
            />
            <PolicyParetoChart
              stats={stats}
              policies={policies}
              logScale={logScale}
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              logMeta={logMeta}
            />
          </div>

          <div className="flex items-center justify-end gap-2">
            <button
              onClick={() => setShowErrorBars((v) => !v)}
              className={`btn-ghost text-xs ${showErrorBars ? "text-accent-secondary" : ""}`}
            >
              {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
            </button>
            <button
              onClick={() => setLogScale((v) => !v)}
              className={`btn-ghost text-xs ${logScale ? "text-accent-secondary" : ""}`}
            >
              {logScale ? "Log scale (on)" : "Log scale (off)"}
            </button>
          </div>

          {/* 4-metric bar charts */}
          <div className="grid grid-cols-2 gap-4">
            <MetricBarChart
              title="Avg Profit by Policy (€) — hover for ±std"
              policies={policies}
              values={metricValues("profit")}
              color="#6366f1"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-profit"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Distance by Policy (km)"
              policies={policies}
              values={metricValues("km")}
              color="#38bdf8"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-km"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Overflows by Policy"
              policies={policies}
              values={metricValues("overflows")}
              color="#f87171"
              logScale={logScale}
              useSymlog
              showErrorBars={showErrorBars}
              exportName="summary-overflows"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Waste Collected by Policy (kg)"
              policies={policies}
              values={metricValues("kg")}
              color="#34d399"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-kg"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
          </div>

          {!logScale && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-300">Log-scale views (§G.1.6)</p>
              <div className="grid grid-cols-2 gap-4">
                <MetricBarChart
                  title="Avg Profit by Policy (€) — symlog scale"
                  policies={policies}
                  values={metricValues("profit")}
                  color="#6366f1"
                  logScale
                  useSymlog
                  exportName="summary-profit-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Distance by Policy (km) — symlog scale"
                  policies={policies}
                  values={metricValues("km")}
                  color="#38bdf8"
                  logScale
                  useSymlog
                  exportName="summary-km-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Overflows by Policy — symlog scale"
                  policies={policies}
                  values={metricValues("overflows")}
                  color="#f87171"
                  logScale
                  useSymlog
                  exportName="summary-overflows-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Waste Collected by Policy (kg) — symlog scale"
                  policies={policies}
                  values={metricValues("kg")}
                  color="#34d399"
                  logScale
                  useSymlog
                  exportName="summary-kg-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
