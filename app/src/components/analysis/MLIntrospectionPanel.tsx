/**
 * ML Introspection — TensorDict/NPZ inspector, attention heatmap, loss contour (§G.5).
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { Download, FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { runTensorArrowPipeline, type ArrowPipelineTiming } from "../../utils/arrowPipeline";
import { buildAttentionGraphOption } from "../../utils/attentionGraph";
import { exportChartPng } from "../../utils/chartExport";
import {
  distributionDisplayName,
  inferDistributionLabel,
} from "../../utils/distributionCompare";
import { GRAPH_PRESETS, loadGraphCoordinates } from "../../utils/graphCoords";
import { chartMetricDisplay, transformMatrixLogScale } from "../../utils/chartLogScale";
import { analyzeLossMinima, resolveBpcMarker, type LandscapeMarker } from "../../utils/lossLandscape";
import {
  CLUSTER_PALETTE,
  reorderRowsByClusters,
  sphericalKMeans,
} from "../../utils/sphericalKMeans";
import {
  applySparseTopK,
  buildMatrixHeatmapOption,
  classifyAttentionRole,
  defaultIndices,
  detectHeadAxis,
  diffMatrices,
  groupAttentionKeys,
  leadingIndexCount,
  suggestAttentionKeys,
  type AttentionRole,
} from "../../utils/tensorHeatmap";
import type { NpzArchiveInfo, NpzVectorData, TensorSlicePreview } from "../../types";

import type { LossLandscapeView } from "./LossLandscape3D";

const LossLandscape3D = lazy(() =>
  import("./LossLandscape3D").then((m) => ({ default: m.LossLandscape3D }))
);
const AttentionSigmaView = lazy(() =>
  import("./AttentionSigmaView").then((m) => ({ default: m.AttentionSigmaView }))
);

type IntrospectionTab = "tensor" | "attention" | "loss";
type CompareMode = "single" | "side-by-side" | "overlay" | "distribution";
type AttentionView = "heatmap" | "graph" | "sigma";

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

function buildLogAwareMatrixHeatmap(
  rawValues: number[][],
  opts: Parameters<typeof buildMatrixHeatmapOption>[1] & { title?: string },
  useLogScale: boolean,
  metricKey = "attention"
): Record<string, unknown> {
  const displayValues = useLogScale
    ? transformMatrixLogScale(rawValues, metricKey, true)
    : rawValues;
  const title = opts.title
    ? `${opts.title}${useLogScale ? " · log colour" : ""}`
    : opts.title;
  const base = buildMatrixHeatmapOption(displayValues, { ...opts, title });
  if (!useLogScale) return base;
  const xLabel = opts.xLabel ?? "Col";
  const yLabel = opts.yLabel ?? "Row";
  return {
    ...base,
    tooltip: {
      position: "top",
      formatter: (p: { value?: [number, number, number] }) => {
        const v = p.value;
        if (!v) return "";
        const raw = rawValues[v[1]]?.[v[0]];
        const rawText = Number.isFinite(raw) ? Number(raw).toFixed(4) : "—";
        return `${yLabel} ${v[1]} · ${xLabel} ${v[0]}<br/>${rawText}`;
      },
    },
  };
}

export function MLIntrospectionPanel({ logScale = false }: { logScale?: boolean }) {
  const { projectRoot, pythonPath, theme } = useAppStore();
  const [tab, setTab] = useState<IntrospectionTab>("tensor");
  const [archivePath, setArchivePath] = useState<string | null>(null);
  const [archive, setArchive] = useState<NpzArchiveInfo | null>(null);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [indices, setIndices] = useState<number[]>([]);
  const [decodeStep, setDecodeStep] = useState(0);
  const [compareStep, setCompareStep] = useState(0);
  const [preview, setPreview] = useState<TensorSlicePreview | null>(null);
  const [comparePreview, setComparePreview] = useState<TensorSlicePreview | null>(null);
  const [distArchivePath, setDistArchivePath] = useState<string | null>(null);
  const [distPreview, setDistPreview] = useState<TensorSlicePreview | null>(null);
  const [lossPreview, setLossPreview] = useState<TensorSlicePreview | null>(null);
  const [lossMarkers, setLossMarkers] = useState<LandscapeMarker[]>([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(32);
  const [sparseK, setSparseK] = useState(0);
  const [compareMode, setCompareMode] = useState<CompareMode>("single");
  const [headIndex, setHeadIndex] = useState(0);
  const [attentionRole, setAttentionRole] = useState<AttentionRole | "all">("all");
  const [attentionView, setAttentionView] = useState<AttentionView>("heatmap");
  const [graphPreset, setGraphPreset] = useState("rm-100");
  const [graphCoords, setGraphCoords] = useState<Array<{ lat: number; lng: number }> | null>(null);
  const [tensorPipeline, setTensorPipeline] = useState<ArrowPipelineTiming | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [clusterK, setClusterK] = useState(0);
  const [lossView, setLossView] = useState<LossLandscapeView>("surface");
  const chartRef = useRef<ReactECharts>(null);
  const compareChartRef = useRef<ReactECharts>(null);
  const graphChartRef = useRef<ReactECharts>(null);
  const lossChartRef = useRef<ReactECharts>(null);

  const pickArchive = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Tensor archives", extensions: ["npz", "npy", "td"] }],
    });
    if (!selected || typeof selected !== "string") return;
    setArchivePath(selected);
  }, []);

  const pickDistArchive = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Tensor archives", extensions: ["npz", "npy", "td"] }],
    });
    if (!selected || typeof selected !== "string") return;
    setDistArchivePath(selected);
  }, []);

  const primaryDistLabel = useMemo(
    () => (archivePath ? distributionDisplayName(inferDistributionLabel(archivePath, archive?.arrays.map((a) => a.key))) : "Primary"),
    [archivePath, archive]
  );

  const distCompareLabel = useMemo(
    () =>
      distArchivePath
        ? distributionDisplayName(inferDistributionLabel(distArchivePath))
        : "Compare",
    [distArchivePath]
  );

  const inspectArchive = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const info = await invoke<NpzArchiveInfo>("inspect_npz_archive", {
        path,
        projectRoot: projectRoot || null,
        pythonExecutable: pythonPath || null,
      });
      setArchive(info);
      const attnKeys = suggestAttentionKeys(info.arrays);
      const lossKey = info.arrays.find((a) => /loss_grid|loss_surface|landscape/i.test(a.key));
      const defaultKey =
        attnKeys[0] ?? lossKey?.key ?? info.arrays.find((a) => a.shape.length >= 2)?.key ?? "";
      setSelectedKey(defaultKey);
      if (defaultKey) {
        const shape = info.arrays.find((a) => a.key === defaultKey)?.shape ?? [];
        setIndices(defaultIndices(shape));
        setHeadIndex(0);
      }
      toast.success("Archive inspected", { description: `${info.arrays.length} array(s)` });
    } catch (err) {
      setArchive(null);
      toast.error("Inspect failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath]);

  useEffect(() => {
    if (archivePath) void inspectArchive(archivePath);
  }, [archivePath, inspectArchive]);

  const selectedArray = archive?.arrays.find((a) => a.key === selectedKey);
  const leading = selectedArray ? leadingIndexCount(selectedArray.shape) : 0;
  const headAxis = selectedArray ? detectHeadAxis(selectedArray.shape, selectedKey) : null;
  const headCount = headAxis != null ? selectedArray?.shape[headAxis] ?? 1 : 0;
  const attentionGroups = useMemo(
    () => (archive ? groupAttentionKeys(archive.arrays) : null),
    [archive]
  );
  const activeAttentionRole = classifyAttentionRole(selectedKey);
  const roleFilteredKeys = useMemo(() => {
    if (!archive || tab !== "attention") return [];
    if (attentionRole === "all") {
      const suggested = suggestAttentionKeys(archive.arrays);
      return suggested.length
        ? suggested
        : archive.arrays.filter((a) => a.shape.length >= 2).map((a) => a.key);
    }
    const group = attentionGroups?.[attentionRole] ?? [];
    return group.length
      ? group
      : archive.arrays.filter((a) => a.shape.length >= 2).map((a) => a.key);
  }, [archive, tab, attentionRole, attentionGroups]);

  const effectiveIndices = useMemo(() => {
    const base = [...indices];
    if (headAxis != null) base[headAxis] = headIndex;
    if (leading >= 1 && tab === "attention") {
      const stepDim = leading - 1;
      if (stepDim >= 0) base[stepDim] = decodeStep;
    }
    return base;
  }, [indices, leading, decodeStep, tab, headAxis, headIndex]);

  const compareIndices = useMemo(() => {
    const base = [...effectiveIndices];
    if (leading >= 1 && tab === "attention" && compareMode !== "single") {
      const stepDim = leading - 1;
      if (stepDim >= 0) base[stepDim] = compareStep;
    }
    return base;
  }, [effectiveIndices, leading, compareStep, tab, compareMode]);

  const loadSlice = useCallback(
    async (path: string, idx: number[], key: string) => {
      if (!path || !key) return null;
      return invoke<TensorSlicePreview>("load_tensor_slice", {
        path,
        key: path.endsWith(".npy") ? null : key,
        indices: idx,
        maxDim: tab === "loss" ? 96 : topK,
        projectRoot: projectRoot || null,
        pythonExecutable: pythonPath || null,
      });
    },
    [tab, topK, projectRoot, pythonPath]
  );

  const loadLossMarkers = useCallback(async (path: string, rows: number, cols: number) => {
    if (!path.endsWith(".npz")) {
      setLossMarkers([]);
      return;
    }
    try {
      const vectors = await invoke<NpzVectorData[]>("load_npz_vectors", {
        path,
        keys: ["theta1", "theta2", "bpc_theta1", "bpc_theta2", "bpc_loss"],
      });
      const bpc = resolveBpcMarker(vectors, rows, cols);
      setLossMarkers(bpc ? [bpc] : []);
    } catch {
      setLossMarkers([]);
    }
  }, []);

  const loadPreview = useCallback(async () => {
    if (!archivePath || !selectedKey) return;
    setLoading(true);
    try {
      const slice = await loadSlice(archivePath, effectiveIndices, selectedKey);
      if (!slice) return;
      if (tab === "loss") {
        setLossPreview(slice);
        const rows = slice.values.length;
        const cols = slice.values[0]?.length ?? 0;
        await loadLossMarkers(archivePath, rows, cols);
      } else {
        setPreview(slice);
        if (tab === "attention" && compareMode === "distribution" && distArchivePath) {
          const dist = await loadSlice(distArchivePath, effectiveIndices, selectedKey);
          setDistPreview(dist);
          setComparePreview(null);
        } else if (tab === "attention" && compareMode !== "single" && compareMode !== "distribution") {
          const cmp = await loadSlice(archivePath, compareIndices, selectedKey);
          setComparePreview(cmp);
          setDistPreview(null);
        } else {
          setComparePreview(null);
          setDistPreview(null);
        }
      }
    } catch (err) {
      toast.error("Slice load failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [
    archivePath,
    selectedKey,
    effectiveIndices,
    compareIndices,
    tab,
    compareMode,
    distArchivePath,
    loadSlice,
    loadLossMarkers,
  ]);

  useEffect(() => {
    if (archivePath && selectedKey) void loadPreview();
  }, [archivePath, selectedKey, effectiveIndices, compareIndices, tab, topK, compareMode, distArchivePath, loadPreview]);

  useEffect(() => {
    if (!archive || tab !== "loss") return;
    const lossKey = archive.arrays.find((a) => /loss_grid|loss_surface|landscape/i.test(a.key))?.key;
    if (lossKey && lossKey !== selectedKey) {
      setSelectedKey(lossKey);
    }
  }, [archive, tab, selectedKey]);

  useEffect(() => {
    if (!projectRoot || tab !== "attention" || (attentionView !== "graph" && attentionView !== "sigma")) return;
    let cancelled = false;
    void loadGraphCoordinates(projectRoot, graphPreset)
      .then((coords) => {
        if (!cancelled) setGraphCoords(coords);
      })
      .catch(() => {
        if (!cancelled) setGraphCoords(null);
      });
    return () => {
      cancelled = true;
    };
  }, [projectRoot, tab, attentionView, graphPreset]);

  const ingestTensorSlice = useCallback(async () => {
    if (!archivePath || !selectedKey) return;
    setPipelineLoading(true);
    try {
      const timing = await runTensorArrowPipeline(archivePath, {
        key: archivePath.endsWith(".npy") ? null : selectedKey,
        indices: effectiveIndices,
        maxDim: topK,
        tableName: "studio_tensor",
        projectRoot,
        pythonExecutable: pythonPath,
      });
      setTensorPipeline(timing);
      toast.success("Tensor slice ingested", {
        description: `${timing.rowCount} rows in ${timing.totalMs} ms`,
      });
    } catch (err) {
      toast.error("DuckDB ingest failed", { description: String(err) });
    } finally {
      setPipelineLoading(false);
    }
  }, [archivePath, selectedKey, effectiveIndices, topK, projectRoot, pythonPath]);

  const clusteredAttention = useMemo(() => {
    if (!preview || tab !== "attention" || clusterK <= 1) {
      return { values: preview?.values ?? null, bandSplits: [] as number[] };
    }
    let grid = preview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    const labels = sphericalKMeans(grid, clusterK);
    const reordered = reorderRowsByClusters(grid, labels);
    return { values: reordered.values, bandSplits: reordered.bandSplits };
  }, [preview, tab, clusterK, sparseK]);

  const processedValues = useMemo(() => {
    if (tab === "attention" && clusterK > 1) return clusteredAttention.values;
    if (!preview) return null;
    let grid = preview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [preview, sparseK, tab, clusterK, clusteredAttention.values]);

  const compareValues = useMemo(() => {
    if (!comparePreview) return null;
    let grid = comparePreview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [comparePreview, sparseK]);

  const distValues = useMemo(() => {
    if (!distPreview) return null;
    let grid = distPreview.values;
    if (sparseK > 0) grid = applySparseTopK(grid, sparseK);
    return grid;
  }, [distPreview, sparseK]);

  const heatmapOption = useMemo(() => {
    if (!processedValues) return null;
    const isDiffMatrix =
      (compareMode === "overlay" && !!compareValues) ||
      (compareMode === "distribution" && !!distValues);
    const values =
      compareMode === "overlay" && compareValues
        ? diffMatrices(processedValues, compareValues)
        : compareMode === "distribution" && distValues
          ? diffMatrices(processedValues, distValues)
          : processedValues;
    const label =
      tab === "attention"
        ? compareMode === "distribution"
          ? `${primaryDistLabel} · ${preview?.key} [${effectiveIndices.join(",")}]${
              distValues ? " Δ" : ""
            }`
          : `Attention · ${preview?.key} [${effectiveIndices.join(",")}]${
              compareMode === "overlay" && compareValues ? " Δ" : ""
            }`
        : `${preview?.key} [${preview?.full_shape.join("×")}]`;
    const heatmapOpts = {
      title: label,
      min: compareMode === "overlay" || compareMode === "distribution" ? undefined : preview?.min,
      max: compareMode === "overlay" || compareMode === "distribution" ? undefined : preview?.max,
      theme,
      xLabel: tab === "attention" ? "Key" : "X",
      yLabel: tab === "attention" ? "Query" : "Y",
      attentionRole: tab === "attention" ? activeAttentionRole : undefined,
      clusterBandSplits: tab === "attention" ? clusteredAttention.bandSplits : undefined,
      clusterPalette: CLUSTER_PALETTE,
    };
    const useLogScale = logScale && tab === "attention" && !isDiffMatrix;
    return buildLogAwareMatrixHeatmap(values, heatmapOpts, useLogScale);
  }, [
    processedValues,
    compareValues,
    distValues,
    compareMode,
    preview,
    tab,
    effectiveIndices,
    theme,
    primaryDistLabel,
    activeAttentionRole,
    clusteredAttention.bandSplits,
    logScale,
  ]);

  const compareHeatmapOption = useMemo(() => {
    if (compareMode === "side-by-side" && compareValues) {
      return buildLogAwareMatrixHeatmap(
        compareValues,
        {
          title: `Compare step ${compareStep}`,
          min: comparePreview?.min,
          max: comparePreview?.max,
          theme,
          xLabel: "Key",
          yLabel: "Query",
          attentionRole: activeAttentionRole,
        },
        logScale
      );
    }
    if (compareMode === "distribution" && distValues) {
      return buildLogAwareMatrixHeatmap(
        distValues,
        {
          title: `${distCompareLabel} · ${distPreview?.key ?? selectedKey}`,
          min: distPreview?.min,
          max: distPreview?.max,
          theme,
          xLabel: "Key",
          yLabel: "Query",
          attentionRole: activeAttentionRole,
        },
        logScale
      );
    }
    return null;
  }, [
    compareValues,
    distValues,
    compareMode,
    compareStep,
    comparePreview,
    distPreview,
    distCompareLabel,
    selectedKey,
    theme,
    activeAttentionRole,
    logScale,
  ]);

  const primaryHeatmapOption = useMemo(() => {
    if (!processedValues || compareMode !== "distribution") return heatmapOption;
    return buildLogAwareMatrixHeatmap(
      processedValues,
      {
        title: `${primaryDistLabel} · ${preview?.key ?? selectedKey} [${effectiveIndices.join(",")}]`,
        min: preview?.min,
        max: preview?.max,
        theme,
        xLabel: "Key",
        yLabel: "Query",
        attentionRole: activeAttentionRole,
      },
      logScale
    );
  }, [
    processedValues,
    compareMode,
    heatmapOption,
    primaryDistLabel,
    preview,
    selectedKey,
    effectiveIndices,
    theme,
    activeAttentionRole,
    logScale,
  ]);

  const attentionGraphOption = useMemo(() => {
    if (!processedValues || !graphCoords?.length || attentionView !== "graph") return null;
    return buildAttentionGraphOption(graphCoords, processedValues, {
      title: `Attention graph · query row ${decodeStep} · ${selectedKey}`,
      theme,
      queryRow: decodeStep,
      topK: sparseK > 0 ? sparseK : 24,
      sparseValues: sparseK > 0 ? processedValues : undefined,
    });
  }, [processedValues, graphCoords, attentionView, decodeStep, selectedKey, theme, sparseK]);

  const lossOption = useMemo(() => {
    if (!lossPreview) return null;
    const rawValues = lossPreview.values;
    const displayValues = logScale
      ? rawValues.map((row) =>
          row.map((v) => (Number.isFinite(v) ? (chartMetricDisplay(v, "loss", true) ?? v) : v))
        )
      : rawValues;
    const base = buildMatrixHeatmapOption(displayValues, {
      title: `Loss landscape · ${lossPreview.key}${logScale ? " · log colour" : ""}`,
      theme,
      xLabel: "θ₁",
      yLabel: "θ₂",
    });
    const withTooltip: Record<string, unknown> = {
      ...base,
      tooltip: {
        position: "top",
        formatter: (p: { value?: [number, number, number] }) => {
          const v = p.value;
          if (!v) return "";
          const raw = rawValues[v[1]]?.[v[0]];
          const rawText = Number.isFinite(raw) ? Number(raw).toFixed(4) : "—";
          return `θ₂ ${v[1]} · θ₁ ${v[0]}<br/>loss ${rawText}`;
        },
      },
    };
    if (!lossMarkers.length) return withTooltip;
    const markPoint = {
      data: lossMarkers.map((m) => ({
        name: m.label,
        coord: [m.col, m.row],
        value: m.loss != null ? m.loss.toFixed(4) : "",
        itemStyle: { color: m.color ?? "#f59e0b" },
      })),
      symbol: "diamond",
      symbolSize: 14,
    };
    const series = (withTooltip.series as Record<string, unknown>[])?.[0];
    if (!series) return withTooltip;
    return { ...withTooltip, series: [{ ...series, markPoint }] };
  }, [lossPreview, lossMarkers, theme, logScale]);

  const lossMinima = useMemo(
    () => (lossPreview ? analyzeLossMinima(lossPreview.values) : null),
    [lossPreview]
  );

  const maxDecodeStep = selectedArray && leading >= 1 ? selectedArray.shape[leading - 1] - 1 : 0;

  if (!projectRoot) {
    return (
      <div className="card text-xs text-canvas-muted">
        Set project root to inspect TensorDict / NPZ archives.
      </div>
    );
  }

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">ML Introspection (§G.5)</h2>
          <p className="text-[10px] text-canvas-muted">
            NPZ/NPY/TensorDict (.td) inspector · attention heatmaps · 3D loss topography
          </p>
        </div>
        <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
          {(["tensor", "attention", "loss"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors capitalize ${
                tab === t
                  ? "bg-accent-primary text-white"
                  : "text-canvas-muted hover:text-gray-200"
              }`}
            >
              {t === "tensor" ? "Archive" : t === "attention" ? "Attention" : "Loss"}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button onClick={() => void pickArchive()} className="btn-primary text-xs flex items-center gap-1">
          <FolderOpen size={12} />
          Open .npz / .npy / .td
        </button>
        {archivePath && (
          <button
            onClick={() => archivePath && void inspectArchive(archivePath)}
            disabled={loading}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Re-inspect
          </button>
        )}
        {archivePath && (
          <span className="text-[10px] text-canvas-muted font-mono truncate max-w-md">
            {archivePath.split("/").slice(-3).join("/")}
          </span>
        )}
      </div>

      {archive && tab === "tensor" && (
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-canvas-border">
                <th className="text-left py-2 px-2 text-canvas-muted">Key</th>
                <th className="text-left py-2 px-2 text-canvas-muted">Shape</th>
                <th className="text-left py-2 px-2 text-canvas-muted">Dtype</th>
                <th className="text-right py-2 px-2 text-canvas-muted">Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border">
              {archive.arrays.map((a) => (
                <tr
                  key={a.key}
                  className={`hover:bg-canvas-hover cursor-pointer ${
                    selectedKey === a.key ? "bg-canvas-hover/60" : ""
                  }`}
                  onClick={() => {
                    setSelectedKey(a.key);
                    setIndices(defaultIndices(a.shape));
                    setDecodeStep(0);
                    setHeadIndex(0);
                  }}
                >
                  <td className="py-2 px-2 font-mono text-gray-300">{a.key}</td>
                  <td className="py-2 px-2 font-mono text-canvas-muted">[{a.shape.join(", ")}]</td>
                  <td className="py-2 px-2 text-canvas-muted">{a.dtype}</td>
                  <td className="py-2 px-2 text-right text-canvas-muted">{formatBytes(a.size_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap items-center gap-2 mt-3">
            <button
              onClick={() => void ingestTensorSlice()}
              disabled={!selectedKey || pipelineLoading}
              className="btn-primary text-xs"
            >
              {pipelineLoading ? "Ingesting…" : "Ingest slice → DuckDB"}
            </button>
            {tensorPipeline && (
              <span className="text-[10px] text-canvas-muted font-mono">
                {tensorPipeline.tableName}: {tensorPipeline.rowCount} rows · {tensorPipeline.totalMs} ms
                {tensorPipeline.withinBudget ? " ✓" : " (budget)"}
              </span>
            )}
          </div>
          {tensorPipeline && (
            <p className="text-[10px] text-canvas-muted mt-1 font-mono">
              SELECT row, col, value FROM &quot;{tensorPipeline.tableName}&quot; ORDER BY value DESC LIMIT 20;
            </p>
          )}
          <p className="text-[10px] text-canvas-muted mt-2">
            Total payload {formatBytes(archive.total_bytes)}
            {archive.used_memmap
              ? " · large .npy/.npz eligible for mmap / decompress slice"
              : ""}
          </p>
        </div>
      )}

      {(tab === "attention" || tab === "loss") && archive && (
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-1.5 text-canvas-muted">
            Array
            <select
              className="select-base text-xs py-0.5 max-w-[200px]"
              value={selectedKey}
              onChange={(e) => {
                const key = e.target.value;
                setSelectedKey(key);
                const shape = archive.arrays.find((a) => a.key === key)?.shape ?? [];
                setIndices(defaultIndices(shape));
                setDecodeStep(0);
                setHeadIndex(0);
              }}
            >
              {(tab === "attention"
                ? roleFilteredKeys
                : archive.arrays
                    .filter((a) => /loss_grid|loss_surface|landscape/i.test(a.key) || a.shape.length === 2)
                    .map((a) => a.key)
              ).map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>

          {tab === "attention" && (
            <>
              <label className="flex items-center gap-1 text-canvas-muted">
                Q/K/V
                <select
                  className="select-base text-xs py-0.5"
                  value={attentionRole}
                  onChange={(e) => {
                    const role = e.target.value as AttentionRole | "all";
                    setAttentionRole(role);
                    if (role !== "all" && attentionGroups?.[role]?.[0]) {
                      const key = attentionGroups[role][0];
                      setSelectedKey(key);
                      const shape = archive.arrays.find((a) => a.key === key)?.shape ?? [];
                      setIndices(defaultIndices(shape));
                      setDecodeStep(0);
                      setHeadIndex(0);
                    }
                  }}
                >
                  <option value="all">All</option>
                  <option value="query">Query ({attentionGroups?.query.length ?? 0})</option>
                  <option value="key">Key ({attentionGroups?.key.length ?? 0})</option>
                  <option value="value">Value ({attentionGroups?.value.length ?? 0})</option>
                  <option value="weights">Weights ({attentionGroups?.weights.length ?? 0})</option>
                </select>
              </label>
              <label className="flex items-center gap-1 text-canvas-muted">
                View
                <select
                  className="select-base text-xs py-0.5"
                  value={attentionView}
                  onChange={(e) => setAttentionView(e.target.value as AttentionView)}
                >
                  <option value="heatmap">Heatmap</option>
                  <option value="graph">Graph on coords (ECharts)</option>
                  <option value="sigma">Graph on coords (Sigma.js)</option>
                </select>
              </label>
              {(attentionView === "graph" || attentionView === "sigma") && (
                <label className="flex items-center gap-1 text-canvas-muted">
                  Graph
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
                </label>
              )}
            </>
          )}

          {tab === "attention" && headCount > 1 && (
            <label className="flex items-center gap-1 text-canvas-muted">
              Head
              <select
                className="select-base text-xs py-0.5 w-14"
                value={headIndex}
                onChange={(e) => setHeadIndex(Number(e.target.value))}
              >
                {Array.from({ length: headCount }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </select>
            </label>
          )}

          {tab === "attention" &&
            leading > 1 &&
            indices
              .slice(0, leading - 1)
              .map((_, dim) =>
                dim === headAxis ? null : (
                  <label key={dim} className="flex items-center gap-1 text-canvas-muted">
                    d{dim}
                    <input
                      type="number"
                      min={0}
                      max={(selectedArray?.shape[dim] ?? 1) - 1}
                      value={indices[dim] ?? 0}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setIndices((prev) => {
                          const next = [...prev];
                          next[dim] = v;
                          return next;
                        });
                      }}
                      className="input-base w-14 text-xs py-0.5"
                    />
                  </label>
                )
              )}

          {tab === "attention" && leading >= 1 && (
            <div className="flex items-center gap-2 flex-1 min-w-[180px]">
              <span className="text-canvas-muted">Decode step</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, maxDecodeStep)}
                value={decodeStep}
                onChange={(e) => setDecodeStep(Number(e.target.value))}
                className="flex-1 accent-accent-primary"
              />
              <span className="font-mono text-gray-300 w-8">{decodeStep}</span>
            </div>
          )}

          {tab === "attention" && (
            <>
              <label className="flex items-center gap-1 text-canvas-muted">
                Compare
                <select
                  className="select-base text-xs py-0.5"
                  value={compareMode}
                  onChange={(e) => setCompareMode(e.target.value as CompareMode)}
                >
                  <option value="single">Single</option>
                  <option value="side-by-side">Side-by-side</option>
                  <option value="overlay">Overlay Δ</option>
                  <option value="distribution">Empirical vs Gamma-3</option>
                </select>
              </label>
              {compareMode === "distribution" && (
                <>
                  <button
                    onClick={() => void pickDistArchive()}
                    className="btn-ghost text-xs flex items-center gap-1"
                  >
                    <FolderOpen size={12} />
                    {distArchivePath ? "Change compare archive" : "Open compare .npz"}
                  </button>
                  {archivePath && (
                    <span className="text-canvas-muted">
                      {primaryDistLabel} vs {distCompareLabel}
                    </span>
                  )}
                </>
              )}
              {compareMode !== "single" && compareMode !== "distribution" && (
                <div className="flex items-center gap-2 min-w-[140px]">
                  <span className="text-canvas-muted">vs step</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, maxDecodeStep)}
                    value={compareStep}
                    onChange={(e) => setCompareStep(Number(e.target.value))}
                    className="flex-1 accent-accent-warning"
                  />
                  <span className="font-mono text-gray-300 w-8">{compareStep}</span>
                </div>
              )}
              <label className="flex items-center gap-1 text-canvas-muted">
                Sparse top-k
                <select
                  className="select-base text-xs py-0.5 w-16"
                  value={sparseK}
                  onChange={(e) => setSparseK(Number(e.target.value))}
                >
                  <option value={0}>Off</option>
                  {[4, 8, 16, 32].map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-1 text-canvas-muted">
                K-means
                <select
                  className="select-base text-xs py-0.5 w-16"
                  value={clusterK}
                  onChange={(e) => setClusterK(Number(e.target.value))}
                >
                  <option value={0}>Off</option>
                  {[2, 3, 4, 6, 8].map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-1 text-canvas-muted">
                Matrix cap
                <select
                  className="select-base text-xs py-0.5 w-16"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                >
                  {[24, 32, 48, 64].map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}

          {preview && tab === "attention" && (
            <span className="text-canvas-muted">
              {preview.rust_ms} ms · downsampled
              {preview.used_memmap
                ? " · mmap slice"
                : preview.used_decompress_slice
                  ? " · decompress slice"
                  : ""}
            </span>
          )}
        </div>
      )}

      {tab === "attention" && attentionView === "graph" && attentionGraphOption && (
        <div className="space-y-2">
          <div className="flex justify-end">
            <button
              className="btn-ghost text-xs flex items-center gap-1"
              onClick={() => exportChartPng(graphChartRef, `attention-graph-${selectedKey}.png`)}
            >
              <Download size={12} />
              Export PNG
            </button>
          </div>
          <ReactECharts ref={graphChartRef} option={attentionGraphOption} style={{ height: 400 }} notMerge />
          <p className="text-[10px] text-canvas-muted">
            Bipartite graph overlay: query node (amber) at decode step {decodeStep}; edge opacity ∝
            attention weight to key nodes on {GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label}.
          </p>
        </div>
      )}

      {tab === "attention" && attentionView === "sigma" && processedValues && graphCoords?.length && (
        <div className="space-y-2">
          <Suspense
            fallback={
              <div className="h-[400px] flex items-center justify-center text-xs text-canvas-muted">
                Loading Sigma.js canvas…
              </div>
            }
          >
            <AttentionSigmaView
              coords={graphCoords}
              values={processedValues}
              queryRow={decodeStep}
              topK={sparseK > 0 ? sparseK : 24}
              sparseValues={sparseK > 0 ? processedValues : undefined}
              theme={theme}
            />
          </Suspense>
          <p className="text-[10px] text-canvas-muted">
            Sigma.js WebGL bipartite graph: ForceAtlas2 layout on{" "}
            {GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label}; query node (amber) at decode step{" "}
            {decodeStep}; edge opacity ∝ attention weight.
          </p>
        </div>
      )}

      {tab === "attention" && attentionView === "heatmap" && heatmapOption && (
        <div className="space-y-2">
          <div className="flex justify-end">
            <button
              className="btn-ghost text-xs flex items-center gap-1"
              onClick={() => exportChartPng(chartRef, `attention-${selectedKey}.png`)}
            >
              <Download size={12} />
              Export PNG
            </button>
          </div>
          <div
            className={
              (compareMode === "side-by-side" || compareMode === "distribution") && compareHeatmapOption
                ? "grid grid-cols-1 lg:grid-cols-2 gap-3"
                : ""
            }
          >
            <ReactECharts
              ref={chartRef}
              option={compareMode === "distribution" ? (primaryHeatmapOption ?? heatmapOption) : heatmapOption}
              style={{ height: 360 }}
              notMerge
            />
            {(compareMode === "side-by-side" || compareMode === "distribution") && compareHeatmapOption && (
              <ReactECharts
                ref={compareChartRef}
                option={compareHeatmapOption}
                style={{ height: 360 }}
                notMerge
              />
            )}
          </div>
          <p className="text-[10px] text-canvas-muted">
            Q/K/V colour palettes ({activeAttentionRole}) · spherical k-means row bands · head selector ·
            sparse top-k · decode-step compare · Empirical vs Gamma-3 (side-by-side / overlay Δ).
            {logScale
              ? " Log-scale colour map on raw weights (overlay Δ stays linear)."
              : " Linear colour map."}{" "}
            Graph-on-coords via ECharts or Sigma.js WebGL (View toggle).
          </p>
        </div>
      )}

      {tab === "loss" && (
        <div className="space-y-2">
          {lossPreview ? (
            <>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-canvas-muted">3D view</span>
                <select
                  className="select-base text-xs py-0.5"
                  value={lossView}
                  onChange={(e) => setLossView(e.target.value as LossLandscapeView)}
                >
                  <option value="surface">Surface mesh</option>
                  <option value="voxels">InstancedMesh voxels</option>
                </select>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <Suspense
                  fallback={
                    <div className="h-[280px] flex items-center justify-center text-xs text-canvas-muted">
                      Loading 3D canvas…
                    </div>
                  }
                >
                  <LossLandscape3D values={lossPreview.values} markers={lossMarkers} view={lossView} />
                </Suspense>
                <div className="space-y-2">
                  <div className="flex justify-end">
                    <button
                      className="btn-ghost text-xs flex items-center gap-1"
                      onClick={() => exportChartPng(lossChartRef, "loss-landscape.png")}
                    >
                      <Download size={12} />
                      Export PNG
                    </button>
                  </div>
                  {lossOption && (
                    <ReactECharts ref={lossChartRef} option={lossOption} style={{ height: 248 }} notMerge />
                  )}
                  <p className="text-[10px] text-canvas-muted">
                    {logScale
                      ? "Log-scale colour map — tooltips show raw loss values"
                      : "Linear colour map — loss grid from NPZ export"}
                  </p>
                </div>
              </div>
              {lossMinima && (
                <p className="text-[10px] text-canvas-muted">
                  Minima annotation: {lossMinima.label} basin (sharpness {lossMinima.sharpness.toFixed(3)}).
                  Flatter minima often generalize better across Empirical vs Gamma-3 distributions.
                  {lossMarkers.length > 0
                    ? " Amber diamond = BPC exact-solver projection on the landscape."
                    : " Bundle bpc_theta1/bpc_theta2 in the NPZ (see export_loss_landscape.py) for the BPC marker."}
                </p>
              )}
            </>
          ) : (
            <p className="text-xs text-canvas-muted">
              Open a loss landscape `.npz` (e.g. from{" "}
              <code className="font-mono">logic/gen/export_loss_landscape.py</code>) containing a{" "}
              <code className="font-mono">loss_grid</code> 2D array.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
