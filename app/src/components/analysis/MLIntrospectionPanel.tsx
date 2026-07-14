/**
 * ML Introspection — TensorDict/NPZ inspector, attention heatmap, loss contour (§G.5).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { Download, FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { exportChartPng } from "../../utils/chartExport";
import {
  buildMatrixHeatmapOption,
  defaultIndices,
  leadingIndexCount,
  suggestAttentionKeys,
} from "../../utils/tensorHeatmap";
import type { NpzArchiveInfo, TensorSlicePreview } from "../../types";

type IntrospectionTab = "tensor" | "attention" | "loss";

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

export function MLIntrospectionPanel() {
  const { projectRoot, theme } = useAppStore();
  const [tab, setTab] = useState<IntrospectionTab>("tensor");
  const [archivePath, setArchivePath] = useState<string | null>(null);
  const [archive, setArchive] = useState<NpzArchiveInfo | null>(null);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [indices, setIndices] = useState<number[]>([]);
  const [decodeStep, setDecodeStep] = useState(0);
  const [preview, setPreview] = useState<TensorSlicePreview | null>(null);
  const [lossPreview, setLossPreview] = useState<TensorSlicePreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(32);
  const chartRef = useRef<ReactECharts>(null);
  const lossChartRef = useRef<ReactECharts>(null);

  const pickArchive = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "NumPy archives", extensions: ["npz", "npy"] }],
    });
    if (!selected || typeof selected !== "string") return;
    setArchivePath(selected);
  }, []);

  const inspectArchive = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const info = await invoke<NpzArchiveInfo>("inspect_npz_archive", { path });
      setArchive(info);
      const attnKeys = suggestAttentionKeys(info.arrays);
      const lossKey = info.arrays.find((a) => /loss_grid|loss_surface|landscape/i.test(a.key));
      const defaultKey =
        attnKeys[0] ?? lossKey?.key ?? info.arrays.find((a) => a.shape.length >= 2)?.key ?? "";
      setSelectedKey(defaultKey);
      if (defaultKey) {
        const shape = info.arrays.find((a) => a.key === defaultKey)?.shape ?? [];
        setIndices(defaultIndices(shape));
      }
      toast.success("Archive inspected", { description: `${info.arrays.length} array(s)` });
    } catch (err) {
      setArchive(null);
      toast.error("Inspect failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (archivePath) void inspectArchive(archivePath);
  }, [archivePath, inspectArchive]);

  const selectedArray = archive?.arrays.find((a) => a.key === selectedKey);
  const leading = selectedArray ? leadingIndexCount(selectedArray.shape) : 0;

  const effectiveIndices = useMemo(() => {
    const base = [...indices];
    if (leading >= 1 && tab === "attention") {
      base[leading - 1] = decodeStep;
    }
    return base;
  }, [indices, leading, decodeStep, tab]);

  const loadPreview = useCallback(async () => {
    if (!archivePath || !selectedKey) return;
    setLoading(true);
    try {
      const slice = await invoke<TensorSlicePreview>("load_tensor_slice", {
        path: archivePath,
        key: archivePath.endsWith(".npy") ? null : selectedKey,
        indices: effectiveIndices,
        maxDim: tab === "loss" ? 96 : topK,
      });
      if (tab === "loss") {
        setLossPreview(slice);
      } else {
        setPreview(slice);
      }
    } catch (err) {
      toast.error("Slice load failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [archivePath, selectedKey, effectiveIndices, tab, topK]);

  useEffect(() => {
    if (archivePath && selectedKey) void loadPreview();
  }, [archivePath, selectedKey, effectiveIndices, tab, topK, loadPreview]);

  useEffect(() => {
    if (!archive || tab !== "loss") return;
    const lossKey = archive.arrays.find((a) => /loss_grid|loss_surface|landscape/i.test(a.key))?.key;
    if (lossKey && lossKey !== selectedKey) {
      setSelectedKey(lossKey);
    }
  }, [archive, tab, selectedKey]);

  const heatmapOption = useMemo(() => {
    if (!preview) return null;
    const label =
      tab === "attention"
        ? `Attention · ${preview.key} [${effectiveIndices.join(",")}]`
        : `${preview.key} [${preview.full_shape.join("×")}]`;
    return buildMatrixHeatmapOption(preview.values, {
      title: label,
      min: preview.min,
      max: preview.max,
      theme,
      xLabel: tab === "attention" ? "Key" : "X",
      yLabel: tab === "attention" ? "Query" : "Y",
    });
  }, [preview, tab, effectiveIndices, theme]);

  const lossOption = useMemo(() => {
    if (!lossPreview) return null;
    return buildMatrixHeatmapOption(lossPreview.values, {
      title: `Loss landscape · ${lossPreview.key}`,
      min: lossPreview.min,
      max: lossPreview.max,
      theme,
      xLabel: "θ₁",
      yLabel: "θ₂",
    });
  }, [lossPreview, theme]);

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
            NPZ/NPY inspector · attention heatmaps · loss landscape contour
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
          Open .npz / .npy
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
          <p className="text-[10px] text-canvas-muted mt-2">
            Total payload {formatBytes(archive.total_bytes)}
            {archive.used_memmap ? " · large .npy eligible for mmap probe" : ""}
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
              }}
            >
              {(tab === "attention"
                ? suggestAttentionKeys(archive.arrays).length
                  ? suggestAttentionKeys(archive.arrays)
                  : archive.arrays.filter((a) => a.shape.length >= 2).map((a) => a.key)
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

          {tab === "attention" && leading > 1 &&
            indices.slice(0, leading - 1).map((_, dim) => (
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
            ))}

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
            <label className="flex items-center gap-1 text-canvas-muted">
              Top-k cap
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
          )}

          {preview && tab === "attention" && (
            <span className="text-canvas-muted">{preview.rust_ms} ms · downsampled</span>
          )}
        </div>
      )}

      {tab === "attention" && heatmapOption && (
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
          <ReactECharts ref={chartRef} option={heatmapOption} style={{ height: 360 }} notMerge />
          <p className="text-[10px] text-canvas-muted">
            Edge opacity ∝ attention magnitude (downsampled matrix). Use decode-step slider for
            sequential decoding timeline (§G.5.3 partial).
          </p>
        </div>
      )}

      {tab === "loss" && (
        <div className="space-y-2">
          {lossOption ? (
            <>
              <div className="flex justify-end">
                <button
                  className="btn-ghost text-xs flex items-center gap-1"
                  onClick={() => exportChartPng(lossChartRef, "loss-landscape.png")}
                >
                  <Download size={12} />
                  Export PNG
                </button>
              </div>
              <ReactECharts ref={lossChartRef} option={lossOption} style={{ height: 360 }} notMerge />
            </>
          ) : (
            <p className="text-xs text-canvas-muted">
              Open a loss landscape `.npz` (e.g. from{" "}
              <code className="font-mono">logic/gen/export_loss_landscape.py</code>) containing a{" "}
              <code className="font-mono">loss_grid</code> 2D array.
            </p>
          )}
          <p className="text-[10px] text-canvas-muted">
            2D ECharts contour heatmap adjacent to future React Three Fiber topography (§G.5.2 partial).
          </p>
        </div>
      )}
    </div>
  );
}
