/**
 * Topological graph analytics panel — ECharts force-directed graph (§G.4).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChevronDown, ChevronUp, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import type { SimDayData } from "../../types";
import {
  buildTopologyFromMatrix,
  loadDistanceMatrixForLog,
  type DistanceMatrixData,
} from "../../utils/graphTopology";

interface Props {
  logPath: string | null;
  projectRoot: string | null;
  simData: SimDayData | null;
  theme: "dark" | "light";
  duckdbProfitRange?: [number, number] | null;
}

export function GraphTopologyPanel({
  logPath,
  projectRoot,
  simData,
  theme,
  duckdbProfitRange = null,
}: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [matrix, setMatrix] = useState<DistanceMatrixData | null>(null);
  const [matrixPath, setMatrixPath] = useState<string | null>(null);
  const [kNeighbors, setKNeighbors] = useState(3);
  const [relayoutOnFilter, setRelayoutOnFilter] = useState(false);
  const [fillMin, setFillMin] = useState(0);
  const [fillMax, setFillMax] = useState(100);
  const [layoutSeed, setLayoutSeed] = useState(0);

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

  const fillRange = useMemo((): [number, number] | null => {
    if (fillMin <= 0 && fillMax >= 100) return null;
    return [fillMin, fillMax];
  }, [fillMin, fillMax]);

  const graphPayload = useMemo(() => {
    if (!matrix) return null;
    return buildTopologyFromMatrix(matrix, simData, {
      kNeighbors,
      relayoutOnFilter,
      fillRange,
      layoutIterations: 60 + (layoutSeed % 3) * 20,
    });
  }, [matrix, simData, kNeighbors, relayoutOnFilter, fillRange, layoutSeed]);

  const chartOption = useMemo(() => {
    if (!graphPayload || !matrix) return null;
    const title = matrixPath
      ? `Topology · ${matrix.nodeIds.length} nodes · k=${kNeighbors}`
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
  }, [graphPayload, matrix, matrixPath, kNeighbors, theme]);

  const tourCount = graphPayload?.nodeMeta.filter((n) => n.onTour).length ?? 0;

  return (
    <div className="card space-y-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-xs font-semibold text-gray-300">Graph Topology (§G.4)</p>
          <p className="text-[10px] text-canvas-muted">
            Distance matrix → k-NN edges · force layout · fill cross-filter
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
                  <input
                    type="checkbox"
                    checked={relayoutOnFilter}
                    onChange={(e) => setRelayoutOnFilter(e.target.checked)}
                    className="accent-accent-primary"
                  />
                  Re-layout on filter
                </label>
                <button
                  onClick={() => setLayoutSeed((s) => s + 1)}
                  className="btn-ghost text-xs"
                >
                  Re-run layout
                </button>
                <span className="text-canvas-muted">
                  {tourCount} on tour · {graphPayload?.edges.length ?? 0} edges
                </span>
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] text-canvas-muted">
                  <span>Fill % cross-filter (node highlight)</span>
                  <span>
                    {fillMin}% – {fillMax}%
                  </span>
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
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-slate-400" /> Idle
                </span>
              </div>

              {chartOption && (
                <ReactECharts option={chartOption} style={{ height: 360 }} notMerge />
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
