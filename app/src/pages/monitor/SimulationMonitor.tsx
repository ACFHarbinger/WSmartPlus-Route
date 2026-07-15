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
 * §G.16 additions (recent passes):
 *   - Day scrubber with ◀/▶ step buttons, "Following" badge, "Latest" reset
 *   - Bin-fill strip chart (bin_state_c percentages, sorted descending, overflow highlight)
 *   - Tour sequence table (stop #, bin ID, fill %, collected, mandatory flags)
 *   - deck.gl tile map: overlay vs side-by-side split when exactly 2 policies visible
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ChevronLeft, ChevronRight, Download, FolderOpen, Pause, Play, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { KpiCard } from "../../components/ui/KpiCard";
import { useSimWatcher } from "../../hooks/useSimWatcher";
import { useAppStore } from "../../store/app";
import { recentFileLabel, useRecentFilesStore } from "../../store/recentFiles";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../../store/sim";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { exportChartPngWithToast, exportChartSvgWithToast } from "../../utils/chartExport";
import { hexToRgb } from "../../utils/colors";
import {
  enrichEntriesWithGraphCoords,
  GRAPH_PRESETS,
  guessGraphPreset,
  loadGraphCoordinates,
} from "../../utils/graphCoords";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
  isLogScaleMetric,
} from "../../utils/chartLogScale";
import { splitVehicleTourIndices, VEHICLE_COLORS_RGB } from "../../utils/vehicleTours";
import { GraphTopologyPanel } from "../../components/analysis/GraphTopologyPanel";
import { SqlQueryPanel } from "../../components/analysis/SqlQueryPanel";
import { formatPipelineTimingBadge, runSimulationArrowPipeline } from "../../utils/arrowPipeline";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import type { BinCoord, DayLogEntry, SimDayData } from "../../types";

const DeckRouteMap = lazy(() => import("../../components/maps/DeckRouteMap"));

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

function fillColor(pct: number): string {
  if (pct >= 100) return "#f87171";
  if (pct >= 80) return "#fbbf24";
  return "#34d399";
}

/** Resolve bin coordinates — use lat/lng when present, else circular layout by index. */
function resolvePositions(
  bins: BinCoord[]
): Map<number, [number, number]> {
  const hasGeo = bins.some((b) => b.lat != null && b.lng != null);
  const map = new Map<number, [number, number]>();
  if (hasGeo) {
    for (const b of bins) {
      if (b.lat != null && b.lng != null) map.set(b.id, [b.lng, b.lat]);
    }
    return map;
  }
  const n = bins.length;
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / Math.max(n, 1);
    map.set(bins[i].id, [Math.cos(angle), Math.sin(angle)]);
  }
  return map;
}

// ── ECharts route map preview (§G.16 — Cartesian fallback when deck.gl unavailable)
function RouteMapChart({ data }: { data: SimDayData }) {
  const chartRef = useRef<ReactECharts>(null);
  const { all_bin_coords, tour_indices, bin_state_c, mandatory } = data;

  const option = useMemo(() => {
    if (!all_bin_coords?.length) return null;

    const positions = resolvePositions(all_bin_coords);
    const mandatorySet = new Set(mandatory ?? []);
    const tourSet = new Set(tour_indices ?? []);

    const idleBins = all_bin_coords
      .filter((b) => b.id >= 0 && !tourSet.has(b.id))
      .map((b) => {
        const pos = positions.get(b.id);
        return pos ? { value: pos, name: `#${b.id}` } : null;
      })
      .filter(Boolean);

    const tourStops = (tour_indices ?? []).map((binId) => {
      const pos = positions.get(binId);
      const fill = bin_state_c?.[binId] ?? 0;
      const pct = Math.min(100, fill * 100);
      return pos
        ? {
            value: pos,
            name: `#${binId}`,
            itemStyle: { color: fillColor(pct) },
            symbolSize: mandatorySet.has(binId) ? 12 : 9,
          }
        : null;
    }).filter(Boolean);

    const depot = all_bin_coords.find((b) => b.id === -1);
    const depotPos = depot ? positions.get(-1) : null;

    const segments = splitVehicleTourIndices(data);
    const vehiclePaths = segments.map((segment, vi) => {
      const pathCoords: [number, number][] = [];
      if (depotPos) pathCoords.push(depotPos);
      for (const binId of segment) {
        const pos = positions.get(binId);
        if (pos) pathCoords.push(pos);
      }
      if (depotPos && pathCoords.length > 1) pathCoords.push(depotPos);
      const [r, g, b] = VEHICLE_COLORS_RGB[vi % VEHICLE_COLORS_RGB.length];
      return {
        name: segments.length > 1 ? `Vehicle ${vi + 1}` : "Route",
        type: "line" as const,
        data: pathCoords,
        lineStyle: { color: `rgb(${r},${g},${b})`, width: 2 },
        symbol: "none",
        z: 1,
      };
    });

    return {
      backgroundColor: "transparent",
      grid: { left: 40, right: 20, top: 20, bottom: 30 },
      xAxis: { type: "value", scale: true, axisLabel: { color: "#9090b0", fontSize: 10 } },
      yAxis: { type: "value", scale: true, axisLabel: { color: "#9090b0", fontSize: 10 } },
      series: [
        ...vehiclePaths,
        {
          name: "Bins",
          type: "scatter",
          data: idleBins,
          symbolSize: 5,
          itemStyle: { color: "#4b5563" },
          z: 2,
        },
        {
          name: "Tour stops",
          type: "scatter",
          data: tourStops,
          z: 3,
        },
        ...(depotPos
          ? [{
              name: "Depot",
              type: "scatter" as const,
              data: [{ value: depotPos, name: "Depot" }],
              symbolSize: 14,
              itemStyle: { color: "#a78bfa" },
              z: 4,
            }]
          : []),
      ],
      tooltip: { trigger: "item" },
      legend: { textStyle: { color: "#9090b0", fontSize: 9 }, top: 0 },
    };
  }, [all_bin_coords, tour_indices, bin_state_c, mandatory]);

  if (!option) return null;

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-canvas-muted">Route Map Preview</p>
        <div className="flex items-center gap-1">
          <button
            className="btn-ghost text-xs flex items-center gap-1"
            onClick={() => exportChartPngWithToast(chartRef, "route-map.png")}
          >
            <Download size={12} />
            PNG
          </button>
          <button
            className="btn-ghost text-xs flex items-center gap-1"
            onClick={() => exportChartSvgWithToast(chartRef, "route-map.svg")}
          >
            SVG
          </button>
        </div>
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 280 }} />
    </div>
  );
}

// ── Metric timeseries chart — supports multi-policy overlay
function MetricTimeseries({
  policySeries,
  metricKey,
  label,
  logScale = false,
}: {
  policySeries: { policy: string; entries: DayLogEntry[]; color: string }[];
  metricKey: string;
  label: string;
  logScale?: boolean;
}) {
  const chartRef = useRef<ReactECharts>(null);
  const allDays = [...new Set(policySeries.flatMap((s) => s.entries.map((e) => e.day)))].sort(
    (a, b) => a - b
  );
  const metricLog = logScale && isLogScaleMetric(metricKey);

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
      if (!e) return null;
      const raw = (e.data as Record<string, number>)[metricKey];
      return chartMetricDisplay(raw, metricKey, logScale);
    }),
  }));

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-canvas-muted">
          {label}
          {metricLog ? " · log" : ""}
        </p>
        <ChartExportButtons
          chartRef={chartRef}
          filenameStem={`${metricKey}-timeseries`}
          size={11}
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: policySeries.length > 1 ? 20 : 10, bottom: 30 },
          xAxis: {
            type: "category",
            data: allDays,
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: {
            type: chartMetricYAxisType(metricKey, logScale),
            logBase: 10,
            axisLabel: { color: "#9090b0", fontSize: 10 },
            minorSplitLine: { show: false },
          },
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

  const {
    policy: selectedPolicy,
    sampleId: selectedSample,
    logScale,
    setPolicy,
    setSampleId,
  } = useGlobalFiltersStore();

  // Keep sim store in sync for legacy consumers (AlgorithmComparison, etc.)
  useEffect(() => {
    setSelectedPolicy(selectedPolicy);
    setSelectedSample(selectedSample);
  }, [selectedPolicy, selectedSample, setSelectedPolicy, setSelectedSample]);

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

  const {
    pendingLogPath,
    setPendingLogPath,
    pendingMapCompare,
    setPendingMapCompare,
    projectRoot,
    theme,
  } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();

  const [activeLogPath, setActiveLogPath] = useState<string | null>(null);
  const [graphPreset, setGraphPreset] = useState(GRAPH_PRESETS[0].id);
  const [loadingGraphCoords, setLoadingGraphCoords] = useState(false);

  const loadLogFile = useCallback(
    async (path: string, watch = true) => {
      reset();
      const historical = await invoke<DayLogEntry[]>("load_simulation_log", { path });
      loadEntries(historical);
      setActiveLogPath(path);
      if (watch) setWatchPath(path);
      pushRecent({ path, label: recentFileLabel(path), kind: "log" });

      if (duckdbReady) {
        setDuckdbLoading(true);
        runSimulationArrowPipeline(path, "monitor_sim")
          .then(setLastPipeline)
          .catch((err) => console.warn("Simulation Arrow pipeline:", err))
          .finally(() => setDuckdbLoading(false));
      }
    },
    [reset, loadEntries, setWatchPath, pushRecent, duckdbReady, setLastPipeline, setDuckdbLoading]
  );

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "JSONL Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    await loadLogFile(path);
  }, [loadLogFile]);

  useEffect(() => {
    if (pendingLogPath) {
      void loadLogFile(pendingLogPath, false);
      setPendingLogPath(null);
    }
  }, [pendingLogPath, setPendingLogPath, loadLogFile]);

  const [showSecondary, setShowSecondary] = useState(false);
  const [showBinFill, setShowBinFill] = useState(true);
  const [showTourTable, setShowTourTable] = useState(false);
  const [showRouteMap, setShowRouteMap] = useState(true);
  const [routeMapMode, setRouteMapMode] = useState<"echarts" | "deckgl">("echarts");
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(1);
  const [mapLayout, setMapLayout] = useState<"overlay" | "split">("overlay");
  const [duckdbProfitRange, setDuckdbProfitRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    if (!activeLogPath || !entries.length) return;
    const guessed = guessGraphPreset(activeLogPath, entries);
    if (guessed) setGraphPreset(guessed);
  }, [activeLogPath, entries]);

  const applyGraphCoords = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to load graph coordinates");
      return;
    }
    if (!entries.length) return;
    setLoadingGraphCoords(true);
    try {
      const sampleIndex = selectedSample ?? 0;
      const table = await loadGraphCoordinates(projectRoot, graphPreset, sampleIndex);
      loadEntries(enrichEntriesWithGraphCoords(entries, table));
      setRouteMapMode("deckgl");
      toast.success("Graph coordinates loaded", {
        description: GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label,
      });
    } catch (err) {
      toast.error("Failed to load graph coordinates", { description: String(err) });
    } finally {
      setLoadingGraphCoords(false);
    }
  }, [projectRoot, entries, graphPreset, selectedSample, loadEntries]);

  const hasGeoCoords = useMemo(
    () => displayEntry?.data.all_bin_coords?.some((b) => b.lat != null && b.lng != null) ?? false,
    [displayEntry]
  );

  const hasBinCoords = useMemo(
    () => (displayEntry?.data.all_bin_coords?.length ?? 0) > 0,
    [displayEntry]
  );

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

  const [mapPolicies, setMapPolicies] = useState<string[]>([]);
  const activeMapPolicies = useMemo(
    () => (mapPolicies.length > 0 ? mapPolicies : policies),
    [mapPolicies, policies]
  );

  const toggleMapPolicy = useCallback(
    (p: string) => {
      setMapPolicies((prev) => {
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

  useEffect(() => {
    if (!pendingMapCompare) return;
    setShowRouteMap(true);
    setMapPolicies(pendingMapCompare.policies);
    setMapLayout(pendingMapCompare.layout);
    if (pendingMapCompare.mapMode) setRouteMapMode(pendingMapCompare.mapMode);
    setPendingMapCompare(null);
  }, [pendingMapCompare, setPendingMapCompare]);

  const mapRoutes = useMemo(() => {
    const dayEntries = entries.filter(
      (e) =>
        e.day === displayDay &&
        (selectedSample === null || e.sample_id === selectedSample) &&
        activeMapPolicies.includes(e.policy)
    );
    return dayEntries
      .filter((e) => e.data.all_bin_coords?.length)
      .map((e) => {
        const idx = policies.indexOf(e.policy);
        return {
          id: `${e.policy}-${e.sample_id}`,
          label: e.policy,
          data: e.data,
          color: hexToRgb(POLICY_COLORS[idx % POLICY_COLORS.length]),
        };
      });
  }, [entries, displayDay, selectedSample, activeMapPolicies, policies]);

  // Build per-policy series for MetricTimeseries
  useEffect(() => {
    if (!isPlaying || dayRange.max <= dayRange.min) return;

    const intervalMs = 800 / playbackSpeed;
    const timer = window.setInterval(() => {
      const current = selectedDay ?? dayRange.max;
      if (current >= dayRange.max) {
        setSelectedDay(dayRange.min);
      } else {
        setSelectedDay(current + 1);
      }
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, playbackSpeed, dayRange.min, dayRange.max, selectedDay, setSelectedDay]);

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
      {entries.length > 0 && <GlobalFilterBar showLogScale />}

      {/* Header controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>

        {watchPath && (
          <span className="flex items-center gap-1.5 text-xs text-canvas-muted truncate max-w-md">
            {isWatching && (
              <RefreshCw size={11} className="animate-spin text-accent-success" />
            )}
            {watchPath.split("/").pop()}
            {duckdbLoading && " · DuckDB ingesting…"}
            {!duckdbLoading && lastPipeline?.tableName === "monitor_sim" && (
              <> · {formatPipelineTimingBadge(lastPipeline)}</>
            )}
          </span>
        )}

        {policies.length > 0 && (
          <select
            className="select-base w-40"
            value={selectedPolicy ?? ""}
            onChange={(e) => setPolicy(e.target.value || null)}
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
              setSampleId(e.target.value ? Number(e.target.value) : null)
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
            <button
              onClick={() => {
                if (isPlaying) {
                  setIsPlaying(false);
                } else {
                  if (selectedDay === null) setSelectedDay(dayRange.min);
                  setIsPlaying(true);
                }
              }}
              className="btn-ghost p-1"
              title={isPlaying ? "Pause day playback" : "Play through simulation days"}
            >
              {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            </button>
            <select
              className="select-base text-[10px] py-0.5 w-14"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value) as 1 | 2 | 4)}
              title="Playback speed"
            >
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
            {selectedDay !== null ? (
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setSelectedDay(null);
                }}
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
          <p className="text-[10px] text-canvas-muted">
            {logScale
              ? "Daily KPI timeseries · log-scale (symlog overflows, log profit/km/kg)"
              : "Daily KPI timeseries · linear scale"}
          </p>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label }) => (
              <MetricTimeseries
                key={key}
                policySeries={policySeries}
                metricKey={key}
                label={label}
                logScale={logScale}
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
            <button
              className="btn-ghost text-xs"
              onClick={() => setShowRouteMap((v) => !v)}
            >
              {showRouteMap ? "Hide" : "Show"} route map
            </button>
            {showRouteMap && policies.length > 1 && (
              <div className="flex items-center gap-1 flex-wrap ml-1">
                <span className="text-[10px] text-canvas-muted">Map:</span>
                {policies.map((p, i) => {
                  const color = POLICY_COLORS[i % POLICY_COLORS.length];
                  const active = activeMapPolicies.includes(p);
                  return (
                    <button
                      key={p}
                      onClick={() => toggleMapPolicy(p)}
                      className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                        active ? "opacity-100" : "opacity-35"
                      }`}
                      style={{ borderColor: color, color: active ? color : undefined }}
                    >
                      {p}
                    </button>
                  );
                })}
              </div>
            )}
            {showRouteMap && mapRoutes.length === 2 && (
              <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5 ml-1">
                {(["overlay", "split"] as const).map((layout) => (
                  <button
                    key={layout}
                    onClick={() => setMapLayout(layout)}
                    className={`text-xs px-2 py-0.5 rounded-md capitalize ${
                      mapLayout === layout
                        ? "bg-accent-primary text-white"
                        : "text-canvas-muted hover:text-gray-200"
                    }`}
                  >
                    {layout}
                  </button>
                ))}
              </div>
            )}
            {showRouteMap && !hasGeoCoords && entries.length > 0 && (
              <div className="flex items-center gap-2 ml-1 flex-wrap">
                <select
                  className="select-base text-xs py-0.5"
                  value={graphPreset}
                  onChange={(e) => setGraphPreset(e.target.value)}
                >
                  {GRAPH_PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
                {activeLogPath && guessGraphPreset(activeLogPath, entries) === graphPreset && (
                  <span className="text-[10px] text-accent-secondary">auto-detected</span>
                )}
                <button
                  onClick={() => void applyGraphCoords()}
                  disabled={loadingGraphCoords}
                  className="btn-ghost text-xs"
                >
                  {loadingGraphCoords ? "Loading…" : "Load graph coords"}
                </button>
              </div>
            )}
            {showRouteMap && hasBinCoords && (
              <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5 ml-1">
                {(["echarts", "deckgl"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setRouteMapMode(m)}
                    className={`text-xs px-2 py-0.5 rounded-md ${
                      routeMapMode === m
                        ? "bg-accent-primary text-white"
                        : "text-canvas-muted hover:text-gray-200"
                    }`}
                  >
                    {m === "echarts" ? "ECharts" : hasGeoCoords ? "Mercator" : "OrbitView"}
                  </button>
                ))}
              </div>
            )}
          </div>

          {showRouteMap && mapRoutes.length > 0 ? (
            routeMapMode === "deckgl" && hasBinCoords ? (
              <Suspense fallback={<p className="text-xs text-canvas-muted">Loading tile map…</p>}>
                {mapLayout === "split" && mapRoutes.length === 2 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {mapRoutes.map((route) => (
                      <div key={route.id} className="space-y-1">
                        <p className="text-xs font-mono text-canvas-muted px-1">{route.label}</p>
                        <DeckRouteMap
                          routes={[route]}
                          animate={isPlaying}
                          playbackSpeed={playbackSpeed}
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <DeckRouteMap
                    routes={mapRoutes}
                    animate={isPlaying}
                    playbackSpeed={playbackSpeed}
                  />
                )}
              </Suspense>
            ) : mapLayout === "split" && mapRoutes.length === 2 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {mapRoutes.map((route) => (
                  <div key={route.id} className="space-y-1">
                    <p className="text-xs font-mono text-canvas-muted px-1">{route.label}</p>
                    <RouteMapChart data={route.data} />
                  </div>
                ))}
              </div>
            ) : displayEntry ? (
              <RouteMapChart data={displayEntry.data} />
            ) : null
          ) : null}
          {showBinFill && <BinFillStrip data={displayEntry.data} />}
          {showTourTable && (
            <TourTable
              data={displayEntry.data}
              day={displayDay}
              policy={displayEntry.policy}
            />
          )}

          <GraphTopologyPanel
            logPath={activeLogPath}
            projectRoot={projectRoot}
            simData={displayEntry?.data ?? null}
            theme={theme}
            duckdbProfitRange={duckdbProfitRange}
            entries={filteredEntries}
            displayDay={displayDay}
            dayRange={dayRange}
            onDaySelect={setSelectedDay}
            logScale={logScale}
          />

          {duckdbReady && lastPipeline?.tableName === "monitor_sim" && (
            <SqlQueryPanel
              tableName={lastPipeline.tableName}
              theme={theme}
              onDaySelect={(day) => setSelectedDay(day)}
              onProfitRange={(min, max) => setDuckdbProfitRange([min, max])}
            />
          )}
        </>
      )}
    </div>
  );
}
