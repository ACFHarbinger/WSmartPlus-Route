/**
 * Simulation Digital Twin — real-time visualization of a running simulation (§G.16).
 *
 * Ports the Streamlit `simulation` mode from:
 *   logic/src/ui/pages/simulation/{kpi,map,charts,bins,tour,summary_sections}.py
 *   logic/src/ui/services/log_parser.py  (stream_log_file)
 *   logic/src/ui/services/simulation_analytics.py
 *
 * Architecture: Rust file-watcher emits sim:day_update events → React updates in <200 ms
 * (replaces Streamlit's time.sleep + st.rerun loop).
 *
 * §G.16 additions in this pass:
 *   - Day scrubber with ◀/▶ step buttons, "Following" badge, "Latest" reset
 *   - Bin-fill strip chart (bin_state_c percentages, sorted descending, overflow highlight)
 *   - Tour sequence table (stop #, bin ID, fill %, collected, mandatory flags)
 */
import { useCallback, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ChevronLeft, ChevronRight, FolderOpen, RefreshCw } from "lucide-react";
import { KpiCard } from "../../components/ui/KpiCard";
import { useSimWatcher } from "../../hooks/useSimWatcher";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../../store/sim";
import type { DayLogEntry, SimDayData } from "../../types";

// ── KPI definitions — mirrors _PRIMARY_KPI_MAP and _SECONDARY_KPI_MAP in kpi.py
const PRIMARY_KPIS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "kg", label: "Waste (kg)", lowerIsBetter: false },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
] as const;

const SECONDARY_KPIS = [
  { key: "ncol", label: "Collections", lowerIsBetter: false },
  { key: "kg_lost", label: "Waste Lost (kg)", lowerIsBetter: true },
  { key: "kg/km", label: "Efficiency (kg/km)", lowerIsBetter: false },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
] as const;

// ── Day-over-day delta
function computeDelta(entries: DayLogEntry[], day: number, key: string): number | null {
  const prev = entries.find((e) => e.day === day - 1);
  const curr = entries.find((e) => e.day === day);
  if (!prev || !curr) return null;
  const a = (curr.data as Record<string, number>)[key];
  const b = (prev.data as Record<string, number>)[key];
  if (typeof a !== "number" || typeof b !== "number") return null;
  return a - b;
}

// 8-colour palette for policy overlay (distinct, readable on dark bg)
const POLICY_COLORS = [
  "#6366f1", "#34d399", "#f87171", "#fbbf24",
  "#a78bfa", "#fb923c", "#38bdf8", "#f472b6",
];

// ── Metric timeseries chart — supports multi-policy overlay
function MetricTimeseries({
  policySeries,
  metricKey,
  label,
}: {
  policySeries: { policy: string; entries: DayLogEntry[]; color: string }[];
  metricKey: string;
  label: string;
}) {
  const allDays = [...new Set(policySeries.flatMap((s) => s.entries.map((e) => e.day)))].sort(
    (a, b) => a - b
  );

  const series = policySeries.map(({ policy, entries, color }) => ({
    name: policy,
    type: "line" as const,
    smooth: true,
    symbol: "circle",
    symbolSize: 3,
    lineStyle: { color, width: 1.8 },
    itemStyle: { color },
    areaStyle: policySeries.length === 1 ? { color: `${color}1e` } : undefined,
    data: allDays.map((day) => {
      const e = entries.find((en) => en.day === day);
      return e ? ((e.data as Record<string, number>)[metricKey] ?? null) : null;
    }),
  }));

  return (
    <div className="card">
      <p className="text-xs text-canvas-muted mb-2">{label}</p>
      <ReactECharts
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: policySeries.length > 1 ? 20 : 10, bottom: 30 },
          xAxis: {
            type: "category",
            data: allDays,
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 10 } },
          legend: policySeries.length > 1
            ? { data: policySeries.map((s) => s.policy), textStyle: { color: "#9090b0", fontSize: 9 }, top: 0 }
            : undefined,
          series,
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 140 }}
      />
    </div>
  );
}

// ── Bin-fill strip chart (mirrors bin_charts.py fill level bars)
function BinFillStrip({ data }: { data: SimDayData }) {
  const { bin_state_c, bin_state_collected, mandatory } = data;
  if (!bin_state_c?.length) return null;

  const mandatorySet = new Set(mandatory ?? []);
  const sorted = [...bin_state_c.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 25);

  return (
    <div className="card space-y-2">
      <p className="text-xs text-canvas-muted">
        Bin Fill Levels — top 25 of {bin_state_c.length} bins
      </p>
      <div className="space-y-1">
        {sorted.map(([idx, fill]) => {
          const pct = Math.min(100, fill * 100);
          const barColor =
            pct >= 100
              ? "bg-accent-danger"
              : pct >= 80
              ? "bg-accent-warning"
              : "bg-accent-success";
          const isMandatory = mandatorySet.has(idx);
          const isCollected = bin_state_collected?.[idx] ?? false;

          return (
            <div key={idx} className="flex items-center gap-2 text-xs">
              <span className="font-mono w-10 text-right text-canvas-muted">#{idx}</span>
              <div className="flex-1 h-2.5 bg-canvas-elevated rounded-full overflow-hidden">
                <div
                  className={`h-full ${barColor} rounded-full`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="font-mono w-10 text-right text-gray-400">
                {pct.toFixed(0)}%
              </span>
              {isMandatory && (
                <span className="text-accent-danger font-bold w-4 text-center" title="Mandatory">!</span>
              )}
              {isCollected && (
                <span className="text-accent-success w-4 text-center" title="Collected">✓</span>
              )}
              {!isMandatory && !isCollected && <span className="w-4" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Tour sequence table (mirrors tour_table in tour.py)
function TourTable({ data, day, policy }: { data: SimDayData; day: number; policy: string }) {
  const { tour, tour_indices, bin_state_c, bin_state_collected, mandatory } = data;

  const indices: number[] = useMemo(() => {
    if (tour_indices?.length) return tour_indices;
    if (tour?.length) {
      return tour.map((stop) => (typeof stop === "number" ? stop : stop.id));
    }
    return [];
  }, [tour, tour_indices]);

  if (indices.length === 0) return null;

  const mandatorySet = new Set(mandatory ?? []);
  const LIMIT = 60;

  return (
    <div className="card space-y-2">
      <p className="text-xs text-canvas-muted">
        Tour — Day {day} · {policy} · {indices.length} stops
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border text-canvas-muted text-left">
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Bin</th>
              <th className="py-1 pr-3 text-right font-normal">Fill</th>
              <th className="py-1 pr-2 text-center font-normal">Collected</th>
              <th className="py-1 text-center font-normal">Mandatory</th>
            </tr>
          </thead>
          <tbody>
            {indices.slice(0, LIMIT).map((binIdx, stop) => {
              const fill = bin_state_c ? bin_state_c[binIdx] : undefined;
              const collected = bin_state_collected?.[binIdx];
              const mandatory = mandatorySet.has(binIdx);
              return (
                <tr
                  key={stop}
                  className={`border-b border-canvas-border/20 hover:bg-canvas-hover ${
                    mandatory ? "text-accent-warning" : "text-gray-300"
                  }`}
                >
                  <td className="py-0.5 pr-3 text-canvas-muted">{stop + 1}</td>
                  <td className="py-0.5 pr-3 font-mono">#{binIdx}</td>
                  <td className="py-0.5 pr-3 text-right font-mono">
                    {fill != null ? `${(fill * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-0.5 pr-2 text-center">
                    {collected ? (
                      <span className="text-accent-success">✓</span>
                    ) : (
                      <span className="text-canvas-muted">—</span>
                    )}
                  </td>
                  <td className="py-0.5 text-center">
                    {mandatory ? (
                      <span className="text-accent-danger font-bold">!</span>
                    ) : (
                      <span className="text-canvas-muted">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {indices.length > LIMIT && (
          <p className="text-xs text-canvas-muted mt-1">
            Showing {LIMIT} of {indices.length} stops.
          </p>
        )}
      </div>
    </div>
  );
}

// ── Main page
export function SimulationMonitor() {
  const {
    entries,
    selectedPolicy,
    selectedSample,
    selectedDay,
    watchPath,
    isWatching,
    loadEntries,
    setSelectedPolicy,
    setSelectedSample,
    setSelectedDay,
    setWatchPath,
    reset,
  } = useSimStore();

  useSimWatcher(watchPath);

  const policies = useMemo(() => uniquePolicies(entries), [entries]);
  const samples = useMemo(() => uniqueSamples(entries), [entries]);

  const filteredEntries = useMemo(
    () => filterEntries(entries, selectedPolicy, selectedSample),
    [entries, selectedPolicy, selectedSample]
  );

  const dayRange = useMemo(() => {
    if (!filteredEntries.length) return { min: 0, max: 0 };
    const days = filteredEntries.map((e) => e.day);
    return { min: Math.min(...days), max: Math.max(...days) };
  }, [filteredEntries]);

  // When selectedDay is null → auto-follow latest (mirrors "Follow latest" in Streamlit)
  const displayDay = selectedDay ?? dayRange.max;
  const displayEntry = filteredEntries.find((e) => e.day === displayDay) ?? null;

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "JSONL Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    reset();
    const historical = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    loadEntries(historical);
    setWatchPath(path);
  }, [reset, loadEntries, setWatchPath]);

  const [showSecondary, setShowSecondary] = useState(false);
  const [showBinFill, setShowBinFill] = useState(true);
  const [showTourTable, setShowTourTable] = useState(false);

  // Chart policy overlay — defaults to all policies; separate from detail-panel selectedPolicy
  const [chartPolicies, setChartPolicies] = useState<string[]>([]);
  const activeChartPolicies = useMemo(
    () => (chartPolicies.length > 0 ? chartPolicies : policies),
    [chartPolicies, policies]
  );

  const toggleChartPolicy = useCallback(
    (p: string) => {
      setChartPolicies((prev) => {
        const active = prev.length > 0 ? prev : policies;
        if (active.includes(p)) {
          const next = active.filter((x) => x !== p);
          return next.length === 0 ? policies : next;
        }
        return [...active, p];
      });
    },
    [policies]
  );

  // Build per-policy series for MetricTimeseries
  const policySeries = useMemo(
    () =>
      activeChartPolicies.map((p, i) => ({
        policy: p,
        color: POLICY_COLORS[i % POLICY_COLORS.length],
        entries: filterEntries(entries, p, selectedSample),
      })),
    [activeChartPolicies, entries, selectedSample]
  );

  return (
    <div className="space-y-4">
      {/* Header controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>

        {watchPath && (
          <span className="flex items-center gap-1.5 text-xs text-canvas-muted truncate max-w-xs">
            {isWatching && (
              <RefreshCw size={11} className="animate-spin text-accent-success" />
            )}
            {watchPath.split("/").pop()}
          </span>
        )}

        {policies.length > 0 && (
          <select
            className="select-base w-40"
            value={selectedPolicy ?? ""}
            onChange={(e) => setSelectedPolicy(e.target.value || null)}
          >
            <option value="">All policies</option>
            {policies.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        )}

        {samples.length > 1 && (
          <select
            className="select-base w-32"
            value={selectedSample ?? ""}
            onChange={(e) =>
              setSelectedSample(e.target.value ? Number(e.target.value) : null)
            }
          >
            <option value="">All samples</option>
            {samples.map((s) => (
              <option key={s} value={s}>Sample {s}</option>
            ))}
          </select>
        )}

        {/* Day scrubber with step buttons and auto-follow */}
        {dayRange.max > 0 && (
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-xs text-canvas-muted">Day</span>
            <button
              onClick={() => setSelectedDay(Math.max(dayRange.min, displayDay - 1))}
              disabled={displayDay <= dayRange.min}
              className="btn-ghost p-1 disabled:opacity-30"
              title="Previous day"
            >
              <ChevronLeft size={12} />
            </button>
            <input
              type="range"
              min={dayRange.min}
              max={dayRange.max}
              value={displayDay}
              onChange={(e) => setSelectedDay(Number(e.target.value))}
              className="w-28 accent-accent-primary"
            />
            <button
              onClick={() => setSelectedDay(Math.min(dayRange.max, displayDay + 1))}
              disabled={displayDay >= dayRange.max}
              className="btn-ghost p-1 disabled:opacity-30"
              title="Next day"
            >
              <ChevronRight size={12} />
            </button>
            <span className="text-xs font-mono text-canvas-muted w-14 text-center">
              {displayDay}/{dayRange.max}
            </span>
            {selectedDay !== null ? (
              <button
                onClick={() => setSelectedDay(null)}
                className="btn-ghost text-xs text-accent-primary"
                title="Jump to latest day and follow live updates"
              >
                Latest ↓
              </button>
            ) : isWatching ? (
              <span className="flex items-center gap-1 text-xs text-accent-success">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
                Following
              </span>
            ) : null}
          </div>
        )}
      </div>

      {/* Policy overlay chip selector — shown when ≥2 policies are present */}
      {policies.length > 1 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-canvas-muted shrink-0">Charts:</span>
          {policies.map((p, i) => {
            const color = POLICY_COLORS[i % POLICY_COLORS.length];
            const active = activeChartPolicies.includes(p);
            return (
              <button
                key={p}
                onClick={() => toggleChartPolicy(p)}
                className={`text-xs px-2 py-0.5 rounded-full border transition-opacity ${
                  active ? "opacity-100" : "opacity-35"
                }`}
                style={{
                  borderColor: color,
                  color: active ? color : undefined,
                  background: active ? `${color}18` : undefined,
                }}
              >
                {p}
              </button>
            );
          })}
        </div>
      )}

      {!displayEntry && (
        <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-3">
          <p className="text-sm">Open a simulation log file to begin monitoring.</p>
          <p className="text-xs">
            The watcher streams new days in real-time as the simulation runs.
          </p>
        </div>
      )}

      {displayEntry && (
        <>
          {/* Primary KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label, lowerIsBetter }) => (
              <KpiCard
                key={key}
                label={label}
                value={(displayEntry.data as Record<string, number>)[key]}
                delta={computeDelta(filteredEntries, displayDay, key)}
                colorize
                lowerIsBetter={lowerIsBetter}
              />
            ))}
          </div>

          {/* Secondary KPIs */}
          <button
            className="btn-ghost text-xs"
            onClick={() => setShowSecondary((v) => !v)}
          >
            {showSecondary ? "Hide" : "Show"} secondary metrics
          </button>
          {showSecondary && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {SECONDARY_KPIS.map(({ key, label, lowerIsBetter }) => (
                <KpiCard
                  key={key}
                  label={label}
                  value={(displayEntry.data as Record<string, number>)[key]}
                  delta={computeDelta(filteredEntries, displayDay, key)}
                  colorize
                  lowerIsBetter={lowerIsBetter}
                />
              ))}
            </div>
          )}

          {/* Timeseries charts — one per primary KPI, each overlaying activeChartPolicies */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label }) => (
              <MetricTimeseries
                key={key}
                policySeries={policySeries}
                metricKey={key}
                label={label}
              />
            ))}
          </div>

          {/* Bin-fill strip chart */}
          <div className="flex items-center gap-2">
            <button
              className="btn-ghost text-xs"
              onClick={() => setShowBinFill((v) => !v)}
            >
              {showBinFill ? "Hide" : "Show"} bin fill chart
            </button>
            <button
              className="btn-ghost text-xs"
              onClick={() => setShowTourTable((v) => !v)}
            >
              {showTourTable ? "Hide" : "Show"} tour table
            </button>
          </div>

          {showBinFill && <BinFillStrip data={displayEntry.data} />}
          {showTourTable && (
            <TourTable
              data={displayEntry.data}
              day={displayDay}
              policy={displayEntry.policy}
            />
          )}
        </>
      )}
    </div>
  );
}
