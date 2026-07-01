/**
 * Training Monitor — live and historical training metrics.
 *
 * Ports the Streamlit `training` mode from:
 *   logic/src/ui/pages/training.py
 *   logic/src/ui/pages/training_charts.py
 *   logic/src/ui/services/data_loader.py (discover_training_runs, load_training_metrics)
 *
 * Training runs are discovered from <projectRoot>/logs/ and their Lightning
 * metrics.csv files are loaded via Rust. Live training progress comes through
 * the process monitor event stream (process:stdout).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { FolderOpen, RefreshCw } from "lucide-react";
import { useAppStore } from "../store/app";
import type { TrainingRun, TrainingMetricsRow } from "../types";

function LossChart({ metrics, runName }: { metrics: TrainingMetricsRow[]; runName: string }) {
  const epochs = metrics.map((r) => r.epoch ?? r.step ?? 0);
  const trainLoss = metrics.map((r) => r.train_loss ?? null);
  const valLoss = metrics.map((r) => r.val_loss ?? null);
  const reward = metrics.map((r) => r.reward ?? null);

  const option = {
    backgroundColor: "transparent",
    legend: {
      data: ["Train Loss", "Val Loss", "Reward"],
      textStyle: { color: "#9090b0", fontSize: 11 },
    },
    grid: { left: 45, right: 20, top: 35, bottom: 30 },
    xAxis: {
      type: "category",
      data: epochs,
      name: "Epoch",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: [
      {
        type: "value",
        name: "Loss",
        nameTextStyle: { color: "#9090b0" },
        axisLabel: { color: "#9090b0", fontSize: 10 },
      },
      {
        type: "value",
        name: "Reward",
        nameTextStyle: { color: "#9090b0" },
        axisLabel: { color: "#9090b0", fontSize: 10 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "Train Loss",
        type: "line",
        data: trainLoss,
        smooth: true,
        lineStyle: { color: "#6366f1", width: 2 },
        itemStyle: { color: "#6366f1" },
        symbol: "none",
      },
      {
        name: "Val Loss",
        type: "line",
        data: valLoss,
        smooth: true,
        lineStyle: { color: "#f87171", width: 2 },
        itemStyle: { color: "#f87171" },
        symbol: "none",
      },
      {
        name: "Reward",
        type: "line",
        yAxisIndex: 1,
        data: reward,
        smooth: true,
        lineStyle: { color: "#34d399", width: 2 },
        itemStyle: { color: "#34d399" },
        symbol: "none",
      },
    ],
    tooltip: { trigger: "axis" },
  };

  return (
    <div className="card">
      <p className="text-sm font-medium text-gray-200 mb-3">{runName}</p>
      <ReactECharts option={option} style={{ height: 240 }} />
    </div>
  );
}

export function TrainingMonitor() {
  const { projectRoot } = useAppStore();
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metricsMap, setMetricsMap] = useState<Record<string, TrainingMetricsRow[]>>({});
  const [loading, setLoading] = useState(false);

  const logsPath = projectRoot ? `${projectRoot}/logs` : "";

  const discover = useCallback(async () => {
    if (!logsPath) return;
    setLoading(true);
    try {
      const found = await invoke<TrainingRun[]>("list_training_runs", {
        logsPath,
      });
      setRuns(found);
    } finally {
      setLoading(false);
    }
  }, [logsPath]);

  useEffect(() => {
    if (logsPath) discover();
  }, [logsPath, discover]);

  const loadMetrics = useCallback(
    async (run: TrainingRun) => {
      if (metricsMap[run.name]) return;
      const rows = await invoke<TrainingMetricsRow[]>("load_training_metrics", {
        runPath: run.path,
      });
      setMetricsMap((m) => ({ ...m, [run.name]: rows }));
    },
    [metricsMap]
  );

  const toggleRun = useCallback(
    (run: TrainingRun) => {
      setSelected((s) =>
        s.includes(run.name) ? s.filter((r) => r !== run.name) : [...s, run.name]
      );
      loadMetrics(run);
    },
    [loadMetrics]
  );

  const selectedRuns = useMemo(
    () => runs.filter((r) => selected.includes(r.name)),
    [runs, selected]
  );

  if (!projectRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <p className="text-sm">Project root not configured.</p>
        <p className="text-xs">Set the project root path in settings to discover training runs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={discover}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          Discover Runs
        </button>
        <span className="text-xs text-canvas-muted">{logsPath}</span>
      </div>

      {runs.length === 0 && !loading && (
        <div className="card text-canvas-muted text-sm">
          No training runs found in <code className="font-mono text-xs">{logsPath}</code>.
        </div>
      )}

      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-3">
            Training Runs
          </p>
          <div className="space-y-1">
            {runs.map((run) => (
              <label
                key={run.name}
                className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-canvas-hover cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selected.includes(run.name)}
                  onChange={() => toggleRun(run)}
                  className="accent-accent-primary"
                />
                <span className="text-sm text-gray-300 font-mono">{run.name}</span>
                <span className="ml-auto flex gap-2 text-xs text-canvas-muted">
                  {run.has_metrics && <span className="text-accent-success">metrics</span>}
                  {run.has_hparams && <span>hparams</span>}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {selectedRuns.map((run) => {
        const metrics = metricsMap[run.name] ?? [];
        return metrics.length > 0 ? (
          <LossChart key={run.name} metrics={metrics} runName={run.name} />
        ) : (
          <div key={run.name} className="card text-canvas-muted text-sm">
            Loading metrics for {run.name}…
          </div>
        );
      })}
    </div>
  );
}
