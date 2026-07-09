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
import { downloadCsv, downloadParquetTable } from "../../utils/tableExport";
import { toast } from "sonner";
import type { DayLogEntry } from "../../types";

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
}: {
  entries: DayLogEntry[];
  policies: string[];
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
      lineStyle: { width: 1.5 },
      color: POLICY_COLORS[i % POLICY_COLORS.length],
      data: allDays.map((day) => {
        const pt = traj[policy]?.find(([d]) => d === day);
        return pt ? pt[1] : null;
      }),
    })),
  }), [traj, allDays, policies]);

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
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
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
            lineStyle: { color: POLICY_COLORS[i % POLICY_COLORS.length] },
            areaStyle: { color: `${POLICY_COLORS[i % POLICY_COLORS.length]}20` },
          })),
        },
      ],
      tooltip: {},
    };
  }, [stats, policies]);

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

function MetricBarChart({
  title,
  policies,
  values,
  color,
  logScale = false,
  showErrorBars = false,
  exportName,
}: {
  title: string;
  policies: string[];
  values: Array<{ mean: number; std: number }>;
  color: string;
  logScale?: boolean;
  showErrorBars?: boolean;
  exportName: string;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ name: string; value: number; dataIndex: number }>)[0];
        const sd = values[p.dataIndex]?.std ?? 0;
        return `${p.name}: ${fmt(p.value, 2)} ± ${fmt(sd, 2)}`;
      },
    },
    xAxis: {
      type: "category" as const,
      data: policies,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 30 },
    },
    yAxis: {
      type: (logScale ? "log" : "value") as "log" | "value",
      logBase: 10,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    grid: { left: 45, right: 10, top: 16, bottom: 55 },
    series: [
      {
        type: "bar" as const,
        data: values.map((v) => (logScale ? Math.max(v.mean, 0.001) : v.mean)),
        itemStyle: { color },
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
  }), [policies, values, color, logScale, showErrorBars]);

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
      <ReactECharts ref={chartRef} option={option} style={{ height: 190 }} />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function SimulationSummary() {
  const { pendingLogPath, setPendingLogPath, projectRoot } = useAppStore();
  const [parquetExporting, setParquetExporting] = useState(false);
  const [logScale, setLogScale] = useState(false);
  const [showErrorBars, setShowErrorBars] = useState(false);
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
          {/* Policy ranking table */}
          <RankingTable
            stats={stats}
            policies={policies}
            onExport={exportRankingCsv}
            onExportParquet={projectRoot ? exportRankingParquet : undefined}
            parquetExporting={parquetExporting}
          />

          {/* Per-day trajectory */}
          <TrajectoryChart entries={filteredEntries} policies={policies} />

          <PolicyRadarChart stats={stats} policies={policies} />

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
              showErrorBars={showErrorBars}
              exportName="summary-profit"
            />
            <MetricBarChart
              title="Avg Distance by Policy (km)"
              policies={policies}
              values={metricValues("km")}
              color="#38bdf8"
              logScale={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-km"
            />
            <MetricBarChart
              title="Avg Overflows by Policy"
              policies={policies}
              values={metricValues("overflows")}
              color="#f87171"
              logScale={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-overflows"
            />
            <MetricBarChart
              title="Avg Waste Collected by Policy (kg)"
              policies={policies}
              values={metricValues("kg")}
              color="#34d399"
              logScale={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-kg"
            />
          </div>
        </>
      )}
    </div>
  );
}
