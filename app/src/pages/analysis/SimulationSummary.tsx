/**
 * Simulation Summary — post-run aggregate analytics (§G.1 / §G.16).
 *
 * Displays:
 *  • Per-policy mean ± std KPI cards
 *  • Policy ranking table (sortable by any metric)
 *  • Per-day trajectory overlay chart (overflows / profit over simulation days)
 *  • Four bar charts: profit, km, overflows, kg/km
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, ChevronUp, ChevronDown } from "lucide-react";
import { useAppStore } from "../../store/app";
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

function RankingTable({ stats, policies }: { stats: Record<string, PolicyStats>; policies: string[] }) {
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
      <p className="text-xs font-semibold text-gray-300 mb-3">Policy Ranking</p>
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
      {allDays.length === 0 ? (
        <p className="text-xs text-canvas-muted">No day data available.</p>
      ) : (
        <ReactECharts option={option} style={{ height: 260 }} />
      )}
    </div>
  );
}

function MetricBarChart({
  title,
  policies,
  values,
  color,
}: {
  title: string;
  policies: string[];
  values: Array<{ mean: number; std: number }>;
  color: string;
}) {
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
    yAxis: { type: "value" as const, axisLabel: { color: "#9090b0", fontSize: 9 } },
    grid: { left: 45, right: 10, top: 16, bottom: 55 },
    series: [
      {
        type: "bar" as const,
        data: values.map((v) => v.mean),
        itemStyle: { color },
        // ECharts error bar using markLine on each bar is not natively supported;
        // std dev is shown in the tooltip instead.
      },
    ],
  }), [policies, values, color]);

  return (
    <div className="card">
      <p className="text-xs text-canvas-muted mb-2">{title}</p>
      <ReactECharts option={option} style={{ height: 190 }} />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function SimulationSummary() {
  const { pendingLogPath, setPendingLogPath } = useAppStore();
  const [entries, setEntries] = useState<DayLogEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);

  const loadLog = useCallback(async (path: string) => {
    const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    setEntries(loaded);
    setLogPath(path);
  }, []);

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

  const stats = useMemo(() => aggregateByPolicy(entries), [entries]);
  const policies = useMemo(() => Object.keys(stats), [stats]);

  const metricValues = (key: MetricKey) =>
    policies.map((p) => ({
      mean: mean(stats[p][key]),
      std: std(stats[p][key]),
    }));

  return (
    <div className="space-y-4">
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
          <RankingTable stats={stats} policies={policies} />

          {/* Per-day trajectory */}
          <TrajectoryChart entries={entries} policies={policies} />

          {/* 4-metric bar charts */}
          <div className="grid grid-cols-2 gap-4">
            <MetricBarChart
              title="Avg Profit by Policy (€) — hover for ±std"
              policies={policies}
              values={metricValues("profit")}
              color="#6366f1"
            />
            <MetricBarChart
              title="Avg Distance by Policy (km)"
              policies={policies}
              values={metricValues("km")}
              color="#38bdf8"
            />
            <MetricBarChart
              title="Avg Overflows by Policy"
              policies={policies}
              values={metricValues("overflows")}
              color="#f87171"
            />
            <MetricBarChart
              title="Avg Waste Collected by Policy (kg)"
              policies={policies}
              values={metricValues("kg")}
              color="#34d399"
            />
          </div>
        </>
      )}
    </div>
  );
}
