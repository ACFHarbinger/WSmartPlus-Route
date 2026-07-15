/**
 * Topological graph analytics panel — ECharts force-directed graph (§G.4).
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChevronDown, ChevronUp, Download, RefreshCw } from "lucide-react";
import { exportChartPng } from "../../utils/chartExport";
import { toast } from "sonner";
import type { DayLogEntry, SimDayData } from "../../types";
import {
  accumulateTourPheromone,
  accumulateTourPheromoneByStep,
  buildTopologyFromMatrix,
  countTourEdgeSteps,
  DENSE_GRAPH_NODE_THRESHOLD,
  loadDistanceMatrixForLog,
  resolveLayoutMode,
  type DistanceMatrixData,
  type TopologyLayoutMode,
  type TopologyNodeMeta,
} from "../../utils/graphTopology";

const TopologySigmaView = lazy(() =>
  import("./TopologySigmaView").then((m) => ({ default: m.TopologySigmaView }))
);
const TopologyCosmographView = lazy(() =>
  import("./TopologyCosmographView").then((m) => ({ default: m.TopologyCosmographView }))
);

type TopologyView = "echarts" | "sigma" | "cosmograph";
type PheromoneTimelineMode = "day" | "iteration";

interface Props {
  logPath: string | null;
  projectRoot: string | null;
  simData: SimDayData | null;
  theme: "dark" | "light";
  duckdbProfitRange?: [number, number] | null;
  entries?: DayLogEntry[];
  displayDay?: number;
  dayRange?: { min: number; max: number };
  syncTimeline?: boolean;
  onDaySelect?: (day: number) => void;
  logScale?: boolean;
}

export function GraphTopologyPanel({
  logPath,
  projectRoot,
  simData,
  theme,
  duckdbProfitRange = null,
  entries = [],
  displayDay = 1,
  dayRange = { min: 1, max: 1 },
  syncTimeline = true,
  onDaySelect,
  logScale = false,
}: Props) {
  const chartRef = useRef<EChartsReact | null>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [matrix, setMatrix] = useState<DistanceMatrixData | null>(null);
  const [matrixPath, setMatrixPath] = useState<string | null>(null);
  const [kNeighbors, setKNeighbors] = useState(3);
  const [relayoutOnFilter, setRelayoutOnFilter] = useState(false);
  const [fillMin, setFillMin] = useState(0);
  const [fillMax, setFillMax] = useState(100);
  const [layoutSeed, setLayoutSeed] = useState(0);
  const [layoutMode, setLayoutMode] = useState<TopologyLayoutMode | "auto">("auto");
  const [showPheromone, setShowPheromone] = useState(false);
  const [pheromoneDay, setPheromoneDay] = useState(displayDay);
  const [pheromoneMode, setPheromoneMode] = useState<PheromoneTimelineMode>("day");
  const [pheromoneStep, setPheromoneStep] = useState(1);
  const [timelineSync, setTimelineSync] = useState(syncTimeline);
  const [topologyView, setTopologyView] = useState<TopologyView>("echarts");

  const loadMatrix = useCallback(async () => {
    if (!logPath) return;
    setLoading(true);
    try {
      const { data, path } = await loadDistanceMatrixForLog(logPath, projectRoot);
      setMatrix(data);
      setMatrixPath(path);
      toast.success("Distance matrix loaded", {
        description: `${data.nodeIds.length} nodes`,
      });
    } catch (err) {
      setMatrix(null);
      setMatrixPath(null);
      toast.error("Failed to load distance matrix", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [logPath, projectRoot]);

  useEffect(() => {
    if (open && logPath && !matrix && !loading) {
      void loadMatrix();
    }
  }, [open, logPath, matrix, loading, loadMatrix]);

  useEffect(() => {
    setTimelineSync(syncTimeline);
  }, [syncTimeline]);

  useEffect(() => {
    if (timelineSync) setPheromoneDay(displayDay);
  }, [displayDay, timelineSync]);

  const fillRange = useMemo((): [number, number] | null => {
    if (fillMin <= 0 && fillMax >= 100) return null;
    return [fillMin, fillMax];
  }, [fillMin, fillMax]);

  const pheromoneDayEntry = useMemo(
    () => entries.find((e) => e.day === pheromoneDay),
    [entries, pheromoneDay]
  );

  const maxTourSteps = useMemo(
    () => countTourEdgeSteps(pheromoneDayEntry, matrix?.nodeIds ?? []),
    [pheromoneDayEntry, matrix]
  );

  useEffect(() => {
    if (pheromoneStep > maxTourSteps) {
      setPheromoneStep(Math.max(1, maxTourSteps));
    }
  }, [maxTourSteps, pheromoneStep]);

  const pheromoneWeights = useMemo(() => {
    if (!showPheromone || !matrix || !entries.length) return new Map<string, number>();
    if (pheromoneMode === "iteration" && pheromoneDayEntry) {
      return accumulateTourPheromoneByStep(
        pheromoneDayEntry,
        matrix.nodeIds,
        pheromoneStep
      );
    }
    return accumulateTourPheromone(entries, matrix.nodeIds, pheromoneDay);
  }, [
    showPheromone,
    matrix,
    entries,
    pheromoneDay,
    pheromoneMode,
    pheromoneDayEntry,
    pheromoneStep,
  ]);

  const resolvedLayout = matrix
    ? resolveLayoutMode(matrix.nodeIds.length, layoutMode)
    : "force";

  const graphPayload = useMemo(() => {
    if (!matrix) return null;
    return buildTopologyFromMatrix(matrix, simData, {
      kNeighbors,
      relayoutOnFilter,
      fillRange,
      layoutIterations: 60 + (layoutSeed % 3) * 20,
      layoutMode,
      pheromoneWeights,
      showPheromone,
      logScale,
      theme,
    });
  }, [
    matrix,
    simData,
    kNeighbors,
    relayoutOnFilter,
    fillRange,
    layoutSeed,
    layoutMode,
    pheromoneWeights,
    showPheromone,
    logScale,
    theme,
  ]);

  const chartOption = useMemo(() => {
    if (!graphPayload || !matrix) return null;
    const layoutLabel = resolvedLayout === "radial" ? "radial dense" : "force";
    const phLabel = showPheromone
      ? pheromoneMode === "iteration"
        ? ` · τ step ${pheromoneStep}/${maxTourSteps || 0} (day ${pheromoneDay})`
        : ` · τ≤day ${pheromoneDay}`
      : "";
    const title = matrixPath
      ? `Topology · ${matrix.nodeIds.length} nodes · k=${kNeighbors} · ${layoutLabel}${phLabel}`
      : undefined;
    return {
      ...graphPayload.option,
      title: title
        ? {
            text: title,
            left: "center",
            top: 4,
            textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" },
          }
        : undefined,
    };
  }, [
    graphPayload,
    matrix,
    matrixPath,
    kNeighbors,
    resolvedLayout,
    showPheromone,
    pheromoneDay,
    pheromoneMode,
    pheromoneStep,
    maxTourSteps,
    theme,
  ]);

  const tourCount = graphPayload?.nodeMeta.filter((n) => n.onTour).length ?? 0;
  const phEdgeCount = showPheromone
    ? [...pheromoneWeights.values()].filter((v) => v > 0).length
    : 0;
  const suggestRadial = (matrix?.nodeIds.length ?? 0) >= DENSE_GRAPH_NODE_THRESHOLD;

  const handlePheromoneDay = (day: number) => {
    setPheromoneDay(day);
    if (timelineSync) onDaySelect?.(day);
  };

  const brushNodeFill = useCallback((meta: TopologyNodeMeta) => {
    if (meta.fillPct == null) return;
    const pad = 5;
    setFillMin(Math.max(0, meta.fillPct - pad));
    setFillMax(Math.min(100, meta.fillPct + pad));
  }, []);

  const clearFillBrush = useCallback(() => {
    setFillMin(0);
    setFillMax(100);
  }, []);

  const topologyClickEvents = useMemo(
    () => ({
      click: (params: { dataType?: string; data?: { id?: string } }) => {
        if (params.dataType !== "node" || params.data?.id == null) return;
        const meta = graphPayload?.nodeMeta[Number(params.data.id)];
        if (meta) brushNodeFill(meta);
      },
    }),
    [graphPayload, brushNodeFill]
  );

  return (
    <div className="card space-y-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-xs font-semibold text-gray-300">Graph Topology (§G.4)</p>
          <p className="text-[10px] text-canvas-muted">
            Distance matrix → k-NN edges · force/radial layout · ACO pheromone trails
            {logScale ? " · log-scale τ" : ""}
          </p>
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <>
          {!logPath && (
            <p className="text-xs text-canvas-muted">Open a simulation log to load the graph.</p>
          )}

          {logPath && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => void loadMatrix()}
                disabled={loading}
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
                {loading ? "Loading…" : "Reload matrix"}
              </button>
              {matrix && (
                <span className="text-[10px] text-canvas-muted font-mono truncate max-w-xs">
                  {matrixPath?.split("/").slice(-3).join("/")}
                </span>
              )}
              {suggestRadial && layoutMode === "auto" && (
                <span className="text-[10px] text-amber-400/90">Auto radial (N≥{DENSE_GRAPH_NODE_THRESHOLD})</span>
              )}
            </div>
          )}

          {matrix && (
            <>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <label className="flex items-center gap-1.5 text-canvas-muted">
                  k-NN
                  <select
                    className="select-base text-xs py-0.5 w-14"
                    value={kNeighbors}
                    onChange={(e) => setKNeighbors(Number(e.target.value))}
                  >
                    {[2, 3, 4, 5].map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-1.5 text-canvas-muted">
                  Layout
                  <select
                    className="select-base text-xs py-0.5 w-24"
                    value={layoutMode}
                    onChange={(e) =>
                      setLayoutMode(e.target.value as TopologyLayoutMode | "auto")
                    }
                  >
                    <option value="auto">Auto</option>
                    <option value="force">Force</option>
                    <option value="radial">Radial dense</option>
                  </select>
                </label>
                <label className="flex items-center gap-1.5 text-canvas-muted">
                  View
                  <select
                    className="select-base text-xs py-0.5 w-36"
                    value={topologyView}
                    onChange={(e) => setTopologyView(e.target.value as TopologyView)}
                  >
                    <option value="echarts">ECharts</option>
                    <option value="sigma">Sigma.js WebGL</option>
                    <option value="cosmograph">Cosmograph (WebGL)</option>
                  </select>
                </label>
                <label className="flex items-center gap-1.5 text-canvas-muted">
                  <input
                    type="checkbox"
                    checked={relayoutOnFilter}
                    onChange={(e) => setRelayoutOnFilter(e.target.checked)}
                    className="accent-accent-primary"
                  />
                  Re-layout on filter
                </label>
                <label className="flex items-center gap-1.5 text-canvas-muted">
                  <input
                    type="checkbox"
                    checked={showPheromone}
                    onChange={(e) => setShowPheromone(e.target.checked)}
                    className="accent-accent-primary"
                  />
                  ACO pheromone trails
                </label>
                <button
                  onClick={() => setLayoutSeed((s) => s + 1)}
                  className="btn-ghost text-xs"
                  disabled={resolvedLayout === "radial"}
                >
                  Re-run layout
                </button>
                {topologyView === "echarts" && chartOption && (
                  <button
                    type="button"
                    onClick={() =>
                      exportChartPng({ current: chartRef.current }, "topology-graph.png")
                    }
                    className="btn-ghost text-xs flex items-center gap-1"
                  >
                    <Download size={12} />
                    PNG
                  </button>
                )}
                <span className="text-canvas-muted">
                  {tourCount} on tour · {graphPayload?.edges.length ?? 0} edges
                  {showPheromone ? ` · ${phEdgeCount} τ edges` : ""}
                </span>
              </div>

              {showPheromone && (
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-canvas-muted">
                    <span>
                      {pheromoneMode === "day"
                        ? "Pheromone timeline (tour edges ≤ day)"
                        : "Tour iteration stepping (τ per ant step)"}
                    </span>
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-1">
                        Mode
                        <select
                          className="select-base text-xs py-0.5 w-24"
                          value={pheromoneMode}
                          onChange={(e) =>
                            setPheromoneMode(e.target.value as PheromoneTimelineMode)
                          }
                        >
                          <option value="day">By day</option>
                          <option value="iteration">By tour step</option>
                        </select>
                      </label>
                      <label className="flex items-center gap-1">
                        <input
                          type="checkbox"
                          checked={timelineSync}
                          onChange={(e) => {
                            const on = e.target.checked;
                            setTimelineSync(on);
                            if (on) setPheromoneDay(displayDay);
                          }}
                          className="accent-accent-primary"
                        />
                        Sync day scrubber
                      </label>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={dayRange.min}
                      max={dayRange.max}
                      value={pheromoneDay}
                      onChange={(e) => handlePheromoneDay(Number(e.target.value))}
                      className="flex-1 accent-amber-500"
                    />
                    <span className="text-xs font-mono text-amber-400/90 w-16 text-right">
                      Day {pheromoneDay}
                    </span>
                  </div>
                  {pheromoneMode === "iteration" && (
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min={maxTourSteps > 0 ? 1 : 0}
                        max={Math.max(maxTourSteps, 1)}
                        value={pheromoneStep}
                        onChange={(e) => setPheromoneStep(Number(e.target.value))}
                        className="flex-1 accent-amber-400"
                        disabled={maxTourSteps === 0}
                      />
                      <span className="text-xs font-mono text-amber-400/90 w-24 text-right">
                        Step {pheromoneStep}/{maxTourSteps}
                      </span>
                    </div>
                  )}
                  <p className="text-[10px] text-canvas-muted">
                    {pheromoneMode === "day"
                      ? "Edge warmth ∝ accumulated τ from consecutive tour stops (decay 0.88/day)."
                      : "Step through tour edge construction; each step deposits τ on the next consecutive stop pair."}
                  </p>
                </div>
              )}

              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] text-canvas-muted">
                  <span>Fill % cross-filter (click node to brush)</span>
                  <div className="flex items-center gap-2">
                    <span>
                      {fillMin}% – {fillMax}%
                    </span>
                    {(fillMin > 0 || fillMax < 100) && (
                      <button onClick={clearFillBrush} className="text-accent-primary hover:underline">
                        Clear
                      </button>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={fillMin}
                    onChange={(e) => setFillMin(Math.min(Number(e.target.value), fillMax))}
                    className="flex-1 accent-accent-primary"
                  />
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={fillMax}
                    onChange={(e) => setFillMax(Math.max(Number(e.target.value), fillMin))}
                    className="flex-1 accent-accent-primary"
                  />
                </div>
                {duckdbProfitRange && (
                  <p className="text-[10px] text-accent-primary">
                    DuckDB profit brush: €{duckdbProfitRange[0].toFixed(0)} – €
                    {duckdbProfitRange[1].toFixed(0)} (day-level; tour nodes follow current day)
                  </p>
                )}
              </div>

              <div className="flex flex-wrap gap-3 text-[10px] text-canvas-muted">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" /> On tour
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-red-400" /> Mandatory
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-indigo-400" /> In fill range
                </span>
                {showPheromone && (
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-amber-400" /> High τ edge
                  </span>
                )}
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-slate-400" /> Idle
                </span>
              </div>

              {graphPayload && topologyView === "sigma" && (
                <Suspense
                  fallback={
                    <div className="h-[360px] flex items-center justify-center text-xs text-canvas-muted">
                      Loading Sigma.js…
                    </div>
                  }
                >
                  <TopologySigmaView
                    nodeMeta={graphPayload.nodeMeta}
                    edges={graphPayload.edges}
                    positions={graphPayload.positions}
                    layoutMode={resolvedLayout}
                    fillRange={fillRange}
                    pheromoneWeights={pheromoneWeights}
                    showPheromone={showPheromone}
                    logScale={logScale}
                    theme={theme}
                    onNodeClick={brushNodeFill}
                  />
                </Suspense>
              )}
              {graphPayload && topologyView === "cosmograph" && (
                <Suspense
                  fallback={
                    <div className="h-[360px] flex items-center justify-center text-xs text-canvas-muted">
                      Loading Cosmograph view…
                    </div>
                  }
                >
                  <TopologyCosmographView
                    nodeMeta={graphPayload.nodeMeta}
                    edges={graphPayload.edges}
                    positions={graphPayload.positions}
                    fillRange={fillRange}
                    pheromoneWeights={pheromoneWeights}
                    showPheromone={showPheromone}
                    logScale={logScale}
                    theme={theme}
                    onNodeClick={brushNodeFill}
                  />
                </Suspense>
              )}
              {chartOption && topologyView === "echarts" && (
                <ReactECharts
                  ref={chartRef}
                  option={chartOption}
                  style={{ height: 360 }}
                  notMerge
                  onEvents={topologyClickEvents}
                />
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
