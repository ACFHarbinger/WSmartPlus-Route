/**
 * HPO Tracker — Optuna study browser with ECharts visualizations (§G.18).
 * Ports Streamlit `hpo_tracker` mode.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { Copy, FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useAppStore } from "../../store/app";
import type { OptunaStudyData, OptunaStudySummary } from "../../types";

const DEFAULT_STORAGE = "sqlite:///assets/hpo/study.db";

const COMPARE_COLORS = ["#6366f1", "#f87171"];

function buildCrossStudyHistoryOption(
  studies: Array<{ name: string; trials: OptunaStudyData["trials"] }>
) {
  const series = studies.map((s, i) => {
    const completed = s.trials.filter((t) => t.state === "COMPLETE" && t.value != null);
    let best = Infinity;
    const bestSoFar = completed.map((t) => {
      best = Math.min(best, t.value as number);
      return best;
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
      type: "value",
      name: "Best objective",
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    series,
    tooltip: { trigger: "axis" },
  };
}

function buildHistoryOption(trials: OptunaStudyData["trials"]) {
  const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
  const numbers = completed.map((t) => t.number);
  const values = completed.map((t) => t.value as number);
  let best = Infinity;
  const bestSoFar = values.map((v) => {
    best = Math.min(best, v);
    return best;
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
      type: "value",
      name: "Objective",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    series: [
      {
        name: "Objective",
        type: "scatter",
        data: values,
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

function buildParallelOption(study: OptunaStudyData) {
  const completed = study.trials.filter((t) => t.state === "COMPLETE" && t.value != null);
  if (completed.length === 0) return null;

  const paramKeys = [
    ...new Set(completed.flatMap((t) => Object.keys(t.params))),
  ].slice(0, 8);

  const schema = [
    { dim: 0, name: "objective", type: "value" },
    ...paramKeys.map((k, i) => ({ dim: i + 1, name: k, type: "value" as const })),
  ];

  const data = completed.map((t) => [
    t.value as number,
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
  const [storageUrl, setStorageUrl] = useState(DEFAULT_STORAGE);
  const [studies, setStudies] = useState<OptunaStudySummary[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<string | null>(null);
  const [studyData, setStudyData] = useState<OptunaStudyData | null>(null);
  const [compareStudy, setCompareStudy] = useState<string | null>(null);
  const [compareStudyData, setCompareStudyData] = useState<OptunaStudyData | null>(null);
  const [loading, setLoading] = useState(false);

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

  const crossStudyOption = useMemo(() => {
    if (!studyData || !compareStudyData) return null;
    return buildCrossStudyHistoryOption([
      { name: studyData.name, trials: studyData.trials },
      { name: compareStudyData.name, trials: compareStudyData.trials },
    ]);
  }, [studyData, compareStudyData]);

  const parallelOption = useMemo(
    () => (studyData ? buildParallelOption(studyData) : null),
    [studyData]
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
        <div className="grid grid-cols-2 gap-4">
          <div className="card">
            <p className="text-xs text-canvas-muted mb-2">Optimisation History</p>
            <ReactECharts
              option={buildHistoryOption(studyData.trials)}
              style={{ height: 260 }}
            />
          </div>
          {Object.keys(studyData.importances).length > 0 && (
            <div className="card">
              <p className="text-xs text-canvas-muted mb-2">Parameter Importance (FANOVA)</p>
              <ReactECharts
                option={buildImportanceOption(studyData.importances)}
                style={{ height: 260 }}
              />
            </div>
          )}
        </div>
      )}

      {crossStudyOption && (
        <div className="card">
          <p className="text-xs text-canvas-muted mb-2">Cross-Study Comparison — Best-So-Far</p>
          <ReactECharts option={crossStudyOption} style={{ height: 280 }} />
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
          <p className="text-xs text-canvas-muted mb-2">Parallel Coordinates</p>
          <ReactECharts option={parallelOption} style={{ height: 320 }} />
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
