/**
 * Data Generation Wizard — full Hydra form for `main.py gen_data` (§G.11).
 *
 * Mirrors the controller justfile gen-data recipe and gen_data.yaml config.
 * Key Hydra args: data.problem, data.data_distributions, data.dataset_type,
 * data.seed, plus per-graph overrides via "Advanced" panel.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { Play, ChevronDown, ChevronUp, Terminal, Activity, CheckCircle, XCircle, FolderOpen, BarChart2 } from "lucide-react";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { toast } from "sonner";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import {
  chartMetricYAxisType,
  displayBarValue,
} from "../../utils/chartLogScale";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useDataGenStore } from "../../store/launchers";
import { useSpawnProcess } from "../../hooks/useSpawnProcess";
import type { DatasetPreviewStats, StdoutLine, StatusUpdate, ProcessStatus } from "../../types";

const PROBLEMS = ["vrpp", "wcvrp", "scwcvrp", "all"] as const;
const DISTRIBUTIONS = ["gamma3", "emp"] as const;
const DATASET_TYPES = [
  { value: "test_simulator", label: "Test Simulator" },
  { value: "train", label: "Training" },
  { value: "train_time", label: "Training (timed)" },
] as const;
const AREAS = ["figueiradafoz", "riomaior"] as const;

const DIST_LABELS: Record<string, string> = {
  gamma3: "Gamma-3",
  emp: "Empirical",
};

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

function DemandHistogram({ option, logScale = false }: { option: object; logScale?: boolean }) {
  const chartRef = useRef<EChartsReact | null>(null);
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs text-canvas-muted">
          Demand distribution{logScale ? " · log-scale counts" : ""}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="dataset-demand-hist"
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 140 }} />
    </div>
  );
}

export function DataGeneration() {
  const { projectRoot, pythonPath, setMode } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const { spawn, launching } = useSpawnProcess();

  // Persisted form state (§D.4 session persistence)
  const {
    dataSource, tsplibPath, sensorCsvPath,
    problem, distributions, datasetType, seed, overwrite,
    area, numLoc, nSamples, nDays, extraOverrides, patch,
  } = useDataGenStore();

  const setDataSource = (v: "synthetic" | "tsplib" | "sensor") => patch({ dataSource: v });
  const setTsplibPath = (v: string) => patch({ tsplibPath: v });
  const setSensorCsvPath = (v: string) => patch({ sensorCsvPath: v });

  const setProblem = (v: string) => patch({ problem: v });
  const setDistributions = (v: string[]) => patch({ distributions: v });
  const setDatasetType = (v: string) => patch({ datasetType: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setOverwrite = (v: boolean) => patch({ overwrite: v });
  const setArea = (v: string) => patch({ area: v });
  const setNumLoc = (v: number) => patch({ numLoc: v });
  const setNSamples = (v: number) => patch({ nSamples: v });
  const setNDays = (v: number) => patch({ nDays: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });

  // Ephemeral UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Live progress state
  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<ProcessStatus | null>(null);
  const [logTail, setLogTail] = useState<string[]>([]);
  const [previewStats, setPreviewStats] = useState<DatasetPreviewStats | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  useEffect(() => {
    if (!liveProcessId) return;
    let unlistenOut: (() => void) | null = null;
    let unlistenStatus: (() => void) | null = null;

    listen<StdoutLine>("process:stdout", (event) => {
      const { id, line } = event.payload;
      if (id !== liveProcessId) return;
      const text = line.startsWith("[stderr]") ? line.slice(8) : line;
      setLogTail((prev) => [...prev.slice(-19), text]);
    }).then((fn) => { unlistenOut = fn; });

    listen<StatusUpdate>("process:status", (event) => {
      if (event.payload.id === liveProcessId) setRunStatus(event.payload.status);
    }).then((fn) => { unlistenStatus = fn; });

    return () => { unlistenOut?.(); unlistenStatus?.(); };
  }, [liveProcessId]);

  const toggleDist = (d: string) => {
    const next = distributions.includes(d)
      ? distributions.filter((x) => x !== d)
      : [...distributions, d];
    setDistributions(next);
  };

  const pickTsplibFile = async () => {
    const path = (await open({
      filters: [{ name: "TSPLIB / VRP", extensions: ["vrp", "tsp"] }],
    })) as string | null;
    if (path) setTsplibPath(path);
  };

  const pickSensorFile = async () => {
    const path = (await open({
      filters: [{ name: "Sensor CSV", extensions: ["csv"] }],
    })) as string | null;
    if (path) setSensorCsvPath(path);
  };

  const previewDataset = async () => {
    if (!projectRoot) return;
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "pt"] }],
    })) as string | null;
    if (!path) return;
    setPreviewLoading(true);
    try {
      const stats = await invoke<DatasetPreviewStats>("preview_dataset_stats", {
        path,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setPreviewStats(stats);
    } catch (err) {
      toast.error("Preview failed", { description: String(err) });
      setPreviewStats(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const demandHistOption = useMemo(() => {
    if (!previewStats?.demand_histogram.length) return null;
    const yKey = "count";
    return {
      backgroundColor: "transparent",
      grid: { left: 40, right: 10, top: 10, bottom: 30 },
      xAxis: {
        type: "category",
        data: ["0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0", "1.0+"],
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: chartMetricYAxisType(yKey, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [{
        type: "bar",
        data: previewStats.demand_histogram.map((v) => displayBarValue(v, yKey, logScale)),
        itemStyle: { color: "#6366f1" },
      }],
      tooltip: { trigger: "axis" },
    };
  }, [previewStats, logScale]);

  const hydraArgs = useMemo(() => {
    const distList = distributions.length > 0 ? distributions.join(",") : "gamma3";
    const sourceArgs =
      dataSource === "tsplib" && tsplibPath
        ? [`data.source=tsplib`, `data.tsplib_instance=${tsplibPath}`]
        : dataSource === "sensor" && sensorCsvPath
        ? [`data.source=sensor`, `data.sensor_file=${sensorCsvPath}`]
        : [
            `data.graphs.0.area=${area}`,
            `data.graphs.0.num_loc=${numLoc}`,
            `data.graphs.0.n_samples=${nSamples}`,
            `data.graphs.0.n_days=${nDays}`,
          ];
    const args = [
      `data.problem=${problem}`,
      `data.data_distributions=[${distList}]`,
      `data.dataset_type=${datasetType}`,
      `data.overwrite=${overwrite}`,
      ...sourceArgs,
      `seed=${seed}`,
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...args, ...extra];
  }, [problem, distributions, datasetType, overwrite, dataSource, tsplibPath, sensorCsvPath, area, numLoc, nSamples, nDays, seed, extraOverrides]);

  const commandPreview = `python main.py gen_data \\\n  ${hydraArgs.join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    if (dataSource === "tsplib" && !tsplibPath.trim()) return;
    if (dataSource === "sensor" && !sensorCsvPath.trim()) return;
    const procId = `gen_data_${Date.now()}`;
    setLiveProcessId(procId);
    setRunStatus(null);
    setLogTail([]);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", "gen_data", ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, dataSource, tsplibPath, sensorCsvPath, hydraArgs, spawn]);

  const dataGenNonce = useLaunchTriggerStore((s) => s.dataGenNonce);
  useEffect(() => {
    if (dataGenNonce > 0) launch();
  }, [dataGenNonce, launch]);

  return (
    <div className="space-y-4 max-w-2xl">
      <GlobalFilterBar showLogScale />

      {/* Data source */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Data Source</h2>
        <div className="flex flex-wrap gap-4">
          {(["synthetic", "tsplib", "sensor"] as const).map((src) => (
            <label key={src} className="flex items-center gap-2 cursor-pointer text-sm text-gray-300">
              <input
                type="radio"
                name="dataSource"
                className="accent-accent-primary"
                checked={dataSource === src}
                onChange={() => setDataSource(src)}
              />
              {src === "synthetic" ? "Synthetic" : src === "tsplib" ? "TSPLIB" : "Sensor CSV"}
            </label>
          ))}
        </div>
        {dataSource === "tsplib" && (
          <div className="flex gap-2">
            <input
              type="text"
              className="input-base font-mono text-xs flex-1"
              value={tsplibPath}
              onChange={(e) => setTsplibPath(e.target.value)}
              placeholder="path/to/instance.vrp"
            />
            <button onClick={pickTsplibFile} className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200">
              <FolderOpen size={13} />
            </button>
          </div>
        )}
        {dataSource === "sensor" && (
          <div className="space-y-1">
            <div className="flex gap-2">
              <input
                type="text"
                className="input-base font-mono text-xs flex-1"
                value={sensorCsvPath}
                onChange={(e) => setSensorCsvPath(e.target.value)}
                placeholder="path/to/bins.csv (timestamp,bin_id,fill_level,waste_type)"
              />
              <button onClick={pickSensorFile} className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200">
                <FolderOpen size={13} />
              </button>
            </div>
            <p className="text-[10px] text-canvas-muted">
              CSV format: timestamp, bin_id, fill_level, waste_type (§12.3 sensor schema)
            </p>
          </div>
        )}
      </div>

      {/* Dataset preview panel */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Instance Preview</h2>
          <button
            onClick={previewDataset}
            disabled={previewLoading || !projectRoot}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <BarChart2 size={12} />
            {previewLoading ? "Loading…" : "Preview .pkl/.pt"}
          </button>
        </div>
        {previewStats ? (
          <div className="space-y-3">
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">Instances</p>
                <p className="font-mono text-gray-100">{previewStats.num_instances}</p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Nodes</p>
                <p className="font-mono text-gray-100">{previewStats.num_nodes ?? "—"}</p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Demand μ±σ</p>
                <p className="font-mono text-gray-100">
                  {previewStats.demand_mean != null
                    ? `${previewStats.demand_mean.toFixed(3)} ± ${(previewStats.demand_std ?? 0).toFixed(3)}`
                    : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">File size</p>
                <p className="font-mono text-gray-100">{formatBytes(previewStats.file_size_bytes)}</p>
              </div>
            </div>
            {demandHistOption && (
              <DemandHistogram option={demandHistOption} logScale={logScale} />
            )}
          </div>
        ) : (
          <p className="text-xs text-canvas-muted">
            Pick a generated dataset to inspect node count, demand statistics, and histogram.
          </p>
        )}
      </div>

      {/* Problem + distribution */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Dataset</h2>

        <div className="flex flex-wrap gap-6">
          {/* Problem */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Problem</label>
            <select
              className="select-base w-36"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
            >
              {PROBLEMS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          {/* Dataset type */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Dataset Type</label>
            <select
              className="select-base w-44"
              value={datasetType}
              onChange={(e) => setDatasetType(e.target.value)}
            >
              {DATASET_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>

          {/* Seed */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Seed</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Distributions */}
        <div>
          <label className="block text-xs text-canvas-muted mb-2">Distributions</label>
          <div className="flex gap-4">
            {DISTRIBUTIONS.map((d) => (
              <label key={d} className="flex items-center gap-2 cursor-pointer text-sm text-gray-300">
                <input
                  type="checkbox"
                  className="accent-accent-primary"
                  checked={distributions.includes(d)}
                  onChange={() => toggleDist(d)}
                />
                {DIST_LABELS[d]}
              </label>
            ))}
          </div>
          {distributions.length === 0 && (
            <p className="text-xs text-accent-warning mt-1">Select at least one distribution.</p>
          )}
        </div>

        {/* Overwrite toggle */}
        <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 w-fit">
          <input
            type="checkbox"
            className="accent-accent-primary"
            checked={overwrite}
            onChange={(e) => setOverwrite(e.target.checked)}
          />
          Overwrite existing files
        </label>
      </div>

      {/* Graph configuration — only for synthetic source */}
      {dataSource === "synthetic" && (
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Graph</h2>
        <div className="flex flex-wrap gap-4">
          {/* Area */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Area</label>
            <select
              className="select-base w-44"
              value={area}
              onChange={(e) => setArea(e.target.value)}
            >
              {AREAS.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </div>

          {/* num_loc */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Locations</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={numLoc}
              min={1}
              onChange={(e) => setNumLoc(Number(e.target.value))}
            />
          </div>

          {/* n_samples */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Samples</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={nSamples}
              min={1}
              onChange={(e) => setNSamples(Number(e.target.value))}
            />
          </div>

          {/* n_days */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Days</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={nDays}
              min={1}
              onChange={(e) => setNDays(Number(e.target.value))}
            />
          </div>
        </div>

        <p className="text-xs text-canvas-muted">
          Configures <code>data.graphs[0]</code>. For multi-graph generation use Advanced Overrides.
        </p>
      </div>
      )}

      {/* Advanced overrides */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Extra Hydra Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <textarea
            className="input-base w-full font-mono text-xs h-20 resize-y mt-3"
            value={extraOverrides}
            onChange={(e) => setExtraOverrides(e.target.value)}
            placeholder="data.penalty_factor=2.5"
            spellCheck={false}
          />
        )}
      </div>

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">Configure Project Root in Settings first.</p>
        )}
        <button
          onClick={launch}
          disabled={
            launching || !projectRoot || distributions.length === 0
            || (dataSource === "tsplib" && !tsplibPath.trim())
            || (dataSource === "sensor" && !sensorCsvPath.trim())
          }
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Generating…" : "Generate Dataset"}
        </button>
      </div>

      {/* Live progress panel */}
      {liveProcessId && (() => {
        const isDone = runStatus !== null && runStatus !== "running";
        return (
          <div className="card space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isDone ? (
                  runStatus === "completed"
                    ? <CheckCircle size={14} className="text-accent-success" />
                    : <XCircle size={14} className="text-accent-danger" />
                ) : (
                  <Activity size={14} className="text-accent-primary animate-pulse" />
                )}
                <h2 className="text-sm font-semibold text-gray-200">
                  {isDone
                    ? runStatus === "completed" ? "Generation Complete" : `Generation ${runStatus}`
                    : "Generating…"}
                </h2>
              </div>
              <button
                onClick={() => setMode("process_monitor")}
                className="btn-ghost text-xs text-canvas-muted"
              >
                Process Monitor
              </button>
            </div>
            {logTail.length > 0 && (
              <div className="bg-canvas-bg rounded-lg p-2 space-y-0.5 max-h-36 overflow-auto">
                {logTail.map((line, i) => (
                  <p key={i} className="text-xs font-mono text-gray-400 leading-snug">{line}</p>
                ))}
              </div>
            )}
            {logTail.length === 0 && !isDone && (
              <p className="text-xs text-canvas-muted">Waiting for output…</p>
            )}
          </div>
        );
      })()}
    </div>
  );
}
