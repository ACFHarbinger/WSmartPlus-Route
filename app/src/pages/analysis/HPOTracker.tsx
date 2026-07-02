/**
 * HPO Tracker — hyperparameter optimization run browser.
 * Ports Streamlit `hpo_tracker` mode.
 * Full Optuna visualization planned in §G.18.
 */
import { useCallback, useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { RefreshCw } from "lucide-react";
import { useAppStore } from "../../store/app";
import type { TrainingRun, TrainingMetricsRow } from "../../types";

export function HPOTracker() {
  const { projectRoot } = useAppStore();
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetricsRow[][]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    try {
      const found = await invoke<TrainingRun[]>("list_training_runs", {
        logsPath: `${projectRoot}/logs`,
      });
      // Load metrics for each run (last checkpoint value as the summary metric)
      const allMetrics = await Promise.all(
        found.map((r) =>
          invoke<TrainingMetricsRow[]>("load_training_metrics", { runPath: r.path })
        )
      );
      setRuns(found);
      setMetrics(allMetrics);
    } finally {
      setLoading(false);
    }
  }, [projectRoot]);

  useEffect(() => {
    if (projectRoot) refresh();
  }, [projectRoot, refresh]);

  // Summary: last reward per run
  const summaryData = runs.map((r, i) => {
    const m = metrics[i] ?? [];
    const last = m[m.length - 1];
    return { name: r.name, reward: last?.reward ?? null };
  });

  const barOption = {
    backgroundColor: "transparent",
    grid: { left: 60, right: 10, top: 20, bottom: 60 },
    xAxis: {
      type: "category",
      data: summaryData.map((d) => d.name),
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 30 },
    },
    yAxis: { type: "value", name: "Final Reward", nameTextStyle: { color: "#9090b0" }, axisLabel: { color: "#9090b0", fontSize: 10 } },
    series: [
      {
        type: "bar",
        data: summaryData.map((d) => d.reward ?? 0),
        itemStyle: { color: "#6366f1" },
      },
    ],
    tooltip: { trigger: "axis" },
  };

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
        Set project root to browse HPO runs.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={refresh}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
        <span className="text-xs text-canvas-muted">{runs.length} run(s)</span>
      </div>

      {summaryData.length > 0 && (
        <div className="card">
          <p className="text-xs text-canvas-muted mb-2">Final Reward by Run</p>
          <ReactECharts option={barOption} style={{ height: 240 }} />
        </div>
      )}

      {summaryData.length === 0 && !loading && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          No HPO runs found. Full Optuna visualization coming in Phase 18.
        </div>
      )}
    </div>
  );
}
