/**
 * HPO Tracker — Optuna study browser with ECharts visualizations (§G.18).
 * Ports Streamlit `hpo_tracker` mode.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { Copy, Download, FolderOpen, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { exportChartPng } from "../../utils/chartExport";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import type { OptunaStudyData, OptunaStudySummary } from "../../types";

const DEFAULT_STORAGE = "sqlite:///assets/hpo/study.db";

const COMPARE_COLORS = ["#6366f1", "#f87171"];

function buildCrossStudyHistoryOption(
  studies: Array<{ name: string; trials: OptunaStudyData["trials"] }>,
  logScale = false
) {
  const series = studies.map((s, i) => {
    const completed = s.trials.filter((t) => t.state === "COMPLETE" && t.value != null);
    let best = Infinity;
    const bestSoFar = completed.map((t) => {
      best = Math.min(best, t.value as number);
      return logScale ? Math.max(best, 1e-8) : best;
    });
    return {
      name: s.name,
      type: "line",
      data: bestSoFar,
      lineStyle: { color: COMPARE_COLORS[i % COMPARE_COLORS.length], width: 2 },
      showSymbol: false,
    };
  });
  const maxLen = Math.max(...series.map((s) => s.data.length), 1);
  return {
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 30, bottom: 40 },
    xAxis: {
      type: "category",
      data: Array.from({ length: maxLen }, (_, i) => i + 1),
      name: "Trial #",
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: logScale ? "Best objective (log)" : "Best objective",
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    series,
    tooltip: { trigger: "axis" },
  };
}

function buildHistoryOption(trials: OptunaStudyData["trials"], logScale = false) {
  const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
  const numbers = completed.map((t) => t.number);
  const values = completed.map((t) => t.value as number);
  const displayValue = (v: number) => (logScale ? Math.max(v, 1e-8) : v);
  let best = Infinity;
  const bestSoFar = values.map((v) => {
    best = Math.min(best, v);
    return displayValue(best);
  });

  return {
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 40 },
    xAxis: {
      type: "category",
      data: numbers,
      name: "Trial",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: logScale ? "Objective (log)" : "Objective",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        name: "Objective",
        type: "scatter",
        data: values.map(displayValue),
        symbolSize: 7,
        itemStyle: { color: "#6366f1" },
      },
      {
        name: "Best so far",
        type: "line",
        data: bestSoFar,
        lineStyle: { color: "#34d399", width: 2 },
        showSymbol: false,
      },
    ],
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    tooltip: { trigger: "axis" },
  };
}

function buildImportanceOption(importances: Record<string, number>) {
  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  return {
    backgroundColor: "transparent",
    grid: { left: 120, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: "value",
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: sorted.map(([k]) => k),
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: sorted.map(([, v]) => v),
        itemStyle: { color: "#818cf8" },
      },
    ],
    tooltip: { trigger: "axis" },
  };
}

function buildParallelOption(study: OptunaStudyData, logScale = false) {
  const completed = study.trials.filter((t) => t.state === "COMPLETE" && t.value != null);
  if (completed.length === 0) return null;

  const paramKeys = [
    ...new Set(completed.flatMap((t) => Object.keys(t.params))),
  ].slice(0, 8);

  const objectiveLabel = logScale ? "objective (log)" : "objective";
  const schema = [
    { dim: 0, name: objectiveLabel, type: "value" },
    ...paramKeys.map((k, i) => ({ dim: i + 1, name: k, type: "value" as const })),
  ];

  const displayObjective = (v: number) => (logScale ? Math.max(v, 1e-8) : v);
  const data = completed.map((t) => [
    displayObjective(t.value as number),
    ...paramKeys.map((k) => {
      const v = t.params[k];
      return typeof v === "number" ? v : 0;
    }),
  ]);

  return {
    backgroundColor: "transparent",
    parallelAxis: schema.map((s, i) => ({
      dim: i,
      name: s.name,
      nameTextStyle: { color: "#9090b0", fontSize: 10 },
      axisLine: { lineStyle: { color: "#2d2d50" } },
    })),
    series: [
      {
        type: "parallel",
        lineStyle: { width: 1, opacity: 0.5, color: "#6366f1" },
        data,
      },
    ],
    tooltip: { trigger: "item" },
  };
}

export function HPOTracker() {
  const { projectRoot, pythonPath } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [storageUrl, setStorageUrl] = useState(DEFAULT_STORAGE);
  const [studies, setStudies] = useState<OptunaStudySummary[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<string | null>(null);
  const [studyData, setStudyData] = useState<OptunaStudyData | null>(null);
  const [compareStudy, setCompareStudy] = useState<string | null>(null);
  const [compareStudyData, setCompareStudyData] = useState<OptunaStudyData | null>(null);
  const [loading, setLoading] = useState(false);
  const historyChartRef = useRef<ReactECharts>(null);
  const importanceChartRef = useRef<ReactECharts>(null);
  const crossStudyChartRef = useRef<ReactECharts>(null);
  const parallelChartRef = useRef<ReactECharts>(null);

  const refreshStudies = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    try {
      const found = await invoke<OptunaStudySummary[]>("list_optuna_studies", {
        storageUrl,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setStudies(found);
      if (found.length > 0 && !selectedStudy) {
        setSelectedStudy(found[0].name);
      }
    } catch (err) {
      toast.error("Failed to list Optuna studies", { description: String(err) });
      setStudies([]);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, storageUrl, selectedStudy]);

  const loadStudy = useCallback(async (name: string) => {
    if (!projectRoot) return;
    setLoading(true);
    try {
      const data = await invoke<OptunaStudyData>("load_optuna_study", {
        storageUrl,
        studyName: name,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setStudyData(data);
    } catch (err) {
      toast.error("Failed to load study", { description: String(err) });
      setStudyData(null);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, storageUrl]);

  useEffect(() => {
    if (projectRoot) refreshStudies();
  }, [projectRoot, storageUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedStudy && projectRoot) loadStudy(selectedStudy);
  }, [selectedStudy]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!compareStudy || !projectRoot) {
      setCompareStudyData(null);
      return;
    }
    invoke<OptunaStudyData>("load_optuna_study", {
      storageUrl,
      studyName: compareStudy,
      projectRoot,
      pythonExecutable: pythonPath || null,
    })
      .then(setCompareStudyData)
      .catch(() => setCompareStudyData(null));
  }, [compareStudy, projectRoot, pythonPath, storageUrl]);

  const historyOption = useMemo(
    () => (studyData ? buildHistoryOption(studyData.trials, logScale) : null),
    [studyData, logScale]
  );

  const crossStudyOption = useMemo(() => {
    if (!studyData || !compareStudyData) return null;
    return buildCrossStudyHistoryOption(
      [
        { name: studyData.name, trials: studyData.trials },
        { name: compareStudyData.name, trials: compareStudyData.trials },
      ],
      logScale
    );
  }, [studyData, compareStudyData, logScale]);

  const parallelOption = useMemo(
    () => (studyData ? buildParallelOption(studyData, logScale) : null),
    [studyData, logScale]
  );

  const copyBestParams = useCallback(async () => {
    if (!studyData?.best_params) return;
    const lines = Object.entries(studyData.best_params).map(
      ([k, v]) => `${k}=${v}`
    );
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      toast.success("Copied best trial params as Hydra overrides");
    } catch {
      toast.error("Clipboard write failed");
    }
  }, [studyData]);

  const pickStorage = async () => {
    const path = (await open({
      filters: [{ name: "SQLite DB", extensions: ["db", "sqlite"] }],
    })) as string | null;
    if (path) setStorageUrl(`sqlite:///${path}`);
  };

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
        Set project root to browse Optuna studies.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale />

      {/* Storage bar */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Optuna Storage</h2>
        <div className="flex gap-2">
          <input
            type="text"
            className="input-base font-mono text-xs flex-1"
            value={storageUrl}
            onChange={(e) => setStorageUrl(e.target.value)}
            placeholder="sqlite:///assets/hpo/study.db"
          />
          <button onClick={pickStorage} className="btn-ghost p-2" title="Browse DB file">
            <FolderOpen size={14} />
          </button>
          <button
            onClick={refreshStudies}
            disabled={loading}
            className="btn-primary flex items-center gap-2"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {studies.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-3">
            <label className="text-xs text-canvas-muted">Study</label>
            <select
              className="select-base flex-1"
              value={selectedStudy ?? ""}
              onChange={(e) => setSelectedStudy(e.target.value)}
            >
              {studies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} — {s.n_complete}/{s.n_trials} trials
                  {s.best_value != null ? ` (best: ${s.best_value.toFixed(4)})` : ""}
                </option>
              ))}
            </select>
            {studyData && Object.keys(studyData.best_params).length > 0 && (
              <button onClick={copyBestParams} className="btn-ghost text-xs flex items-center gap-1">
                <Copy size={12} />
                Copy best params
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-canvas-muted">Compare with</label>
            <select
              className="select-base flex-1 text-xs"
              value={compareStudy ?? ""}
              onChange={(e) => setCompareStudy(e.target.value || null)}
            >
              <option value="">— none —</option>
              {studies
                .filter((s) => s.name !== selectedStudy)
                .map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                    {s.best_value != null ? ` (best: ${s.best_value.toFixed(4)})` : ""}
                  </option>
                ))}
            </select>
          </div>

          {studyData && (
            <div className="grid grid-cols-4 gap-3 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">Trials</p>
                <p className="text-lg font-mono text-gray-100">{studyData.trials.length}</p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Completed</p>
                <p className="text-lg font-mono text-gray-100">
                  {studyData.trials.filter((t) => t.state === "COMPLETE").length}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Best Value</p>
                <p className="text-lg font-mono text-accent-success">
                  {studyData.best_value != null ? studyData.best_value.toFixed(6) : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">Parameters</p>
                <p className="text-lg font-mono text-gray-100">
                  {Object.keys(studyData.importances).length || "—"}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {studyData && studyData.trials.filter((t) => t.state === "COMPLETE").length >= 2 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-canvas-muted">Optimisation History</p>
              <button
                className="btn-ghost p-0.5"
                title="Export PNG"
                onClick={() => exportChartPng(historyChartRef, "hpo-history.png")}
              >
                <Download size={11} className="text-canvas-muted" />
              </button>
            </div>
            <p className="text-[10px] text-canvas-muted mb-1">
              {logScale
                ? "Log-scale objective axis — trial scatter + best-so-far line"
                : "Linear objective — trial scatter + best-so-far line"}
            </p>
            <ReactECharts
              ref={historyChartRef}
              option={historyOption ?? {}}
              style={{ height: 260 }}
            />
          </div>
          {Object.keys(studyData.importances).length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">Parameter Importance (FANOVA)</p>
                <button
                  className="btn-ghost p-0.5"
                  title="Export PNG"
                  onClick={() => exportChartPng(importanceChartRef, "hpo-importance.png")}
                >
                  <Download size={11} className="text-canvas-muted" />
                </button>
              </div>
              <ReactECharts
                ref={importanceChartRef}
                option={buildImportanceOption(studyData.importances)}
                style={{ height: 260 }}
              />
            </div>
          )}
        </div>
      )}

      {crossStudyOption && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">Cross-Study Comparison — Best-So-Far</p>
            <button
              className="btn-ghost p-0.5"
              title="Export PNG"
              onClick={() => exportChartPng(crossStudyChartRef, "hpo-cross-study.png")}
            >
              <Download size={11} className="text-canvas-muted" />
            </button>
          </div>
          <ReactECharts ref={crossStudyChartRef} option={crossStudyOption} style={{ height: 280 }} />
          {compareStudyData && studyData && (
            <div className="grid grid-cols-2 gap-3 mt-3 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">{studyData.name}</p>
                <p className="font-mono text-accent-success">
                  {studyData.best_value != null ? studyData.best_value.toFixed(6) : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">{compareStudyData.name}</p>
                <p className="font-mono text-accent-danger">
                  {compareStudyData.best_value != null ? compareStudyData.best_value.toFixed(6) : "—"}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {parallelOption && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">Parallel Coordinates</p>
            <button
              className="btn-ghost p-0.5"
              title="Export PNG"
              onClick={() => exportChartPng(parallelChartRef, "hpo-parallel.png")}
            >
              <Download size={11} className="text-canvas-muted" />
            </button>
          </div>
          <p className="text-[10px] text-canvas-muted mb-1">
            {logScale
              ? "Log-scale objective axis — hyperparameter polylines"
              : "Linear objective — hyperparameter polylines"}
          </p>
          <ReactECharts ref={parallelChartRef} option={parallelOption} style={{ height: 320 }} />
        </div>
      )}

      {studies.length === 0 && !loading && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          No Optuna studies found. Check the storage URL or run an HPO sweep first.
        </div>
      )}
    </div>
  );
}
