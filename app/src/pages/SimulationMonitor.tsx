/**
 * Simulation Digital Twin — real-time visualization of a running simulation.
 *
 * Ports the Streamlit `simulation` mode from:
 *   logic/src/ui/pages/simulation/{kpi,map,charts,bins,tour,summary_sections}.py
 *   logic/src/ui/services/log_parser.py  (stream_log_file)
 *   logic/src/ui/services/simulation_analytics.py
 *
 * Key architectural difference from Streamlit:
 *   Streamlit: time.sleep(N) + st.rerun() → full Python script re-execution every N s
 *   Studio: Rust file-watcher emits sim:day_update events → React updates in < 200 ms
 */
import { useCallback, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, RefreshCw } from "lucide-react";
import { KpiCard } from "../components/ui/KpiCard";
import { useSimWatcher } from "../hooks/useSimWatcher";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../store/sim";
import type { DayLogEntry } from "../types";

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

// ── Day-over-day delta helper
function computeDelta(
  entries: DayLogEntry[],
  currentDay: number,
  metricKey: string
): number | null {
  const prev = entries.find((e) => e.day === currentDay - 1);
  const curr = entries.find((e) => e.day === currentDay);
  if (!prev || !curr) return null;
  const a = (curr.data as Record<string, number>)[metricKey];
  const b = (prev.data as Record<string, number>)[metricKey];
  if (typeof a !== "number" || typeof b !== "number") return null;
  return a - b;
}

// ── Multi-day metric chart (mirrors create_sparkline_svg in charts.py)
function MetricTimeseries({
  entries,
  metricKey,
  label,
}: {
  entries: DayLogEntry[];
  metricKey: string;
  label: string;
}) {
  const days = entries.map((e) => e.day);
  const values = entries.map(
    (e) => ((e.data as Record<string, number>)[metricKey] ?? null)
  );

  const option = {
    backgroundColor: "transparent",
    grid: { left: 40, right: 10, top: 10, bottom: 30 },
    xAxis: { type: "category", data: days, axisLabel: { color: "#9090b0", fontSize: 10 } },
    yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 10 } },
    series: [
      {
        type: "line",
        data: values,
        smooth: true,
        lineStyle: { color: "#6366f1", width: 2 },
        areaStyle: { color: "rgba(99,102,241,0.12)" },
        symbol: "circle",
        symbolSize: 4,
        itemStyle: { color: "#818cf8" },
      },
    ],
    tooltip: { trigger: "axis" },
  };

  return (
    <div className="card">
      <p className="text-xs text-canvas-muted mb-2">{label}</p>
      <ReactECharts option={option} style={{ height: 120 }} />
    </div>
  );
}

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

  // Wire up the Rust file-watcher
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

  const displayDay = selectedDay ?? dayRange.max;
  const displayEntry = filteredEntries.find((e) => e.day === displayDay) ?? null;

  // ── File picker
  const openLog = useCallback(async () => {
    const path = await open({
      filters: [{ name: "JSONL Logs", extensions: ["jsonl", "log", "txt"] }],
    }) as string | null;
    if (!path) return;

    reset();
    // Load historical entries first
    const historical = await invoke<DayLogEntry[]>("load_simulation_log", {
      path,
    });
    loadEntries(historical);
    setWatchPath(path);
  }, [reset, loadEntries, setWatchPath]);

  const [showSecondary, setShowSecondary] = useState(false);

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

        {dayRange.max > 0 && (
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-canvas-muted">Day</span>
            <input
              type="range"
              min={dayRange.min}
              max={dayRange.max}
              value={displayDay}
              onChange={(e) => setSelectedDay(Number(e.target.value))}
              className="w-32 accent-accent-primary"
            />
            <span className="text-xs font-mono w-8 text-right">
              {displayDay}/{dayRange.max}
            </span>
          </div>
        )}
      </div>

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
          {/* Primary KPIs — mirrors render_kpi_dashboard in kpi.py */}
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

          {/* Secondary KPIs toggle */}
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

          {/* Timeseries charts — mirrors per-day metric history in simulation_analytics.py */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label }) => (
              <MetricTimeseries
                key={key}
                entries={filteredEntries}
                metricKey={key}
                label={label}
              />
            ))}
          </div>

          {/* Tour info */}
          {displayEntry.data.tour && (
            <div className="card">
              <p className="text-xs text-canvas-muted mb-2">
                Tour — Day {displayDay} · {displayEntry.policy}
              </p>
              <p className="text-sm text-gray-300">
                <span className="font-medium">{displayEntry.data.tour.length}</span> stops ·{" "}
                <span className="font-medium">
                  {(displayEntry.data.bin_state_collected ?? []).filter(Boolean).length}
                </span>{" "}
                collected
              </p>
              <p className="text-xs text-canvas-muted mt-2">
                Route map (deck.gl integration) coming in Phase 3.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
