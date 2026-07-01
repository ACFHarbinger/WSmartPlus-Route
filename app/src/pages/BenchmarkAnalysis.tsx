/**
 * Benchmark Analysis — multi-run, multi-policy comparison.
 * Ports Streamlit `benchmark` mode.
 */
import { useCallback, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, X } from "lucide-react";
import type { DayLogEntry } from "../types";

interface RunFile {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

const METRICS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
];

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

export function BenchmarkAnalysis() {
  const [runs, setRuns] = useState<RunFile[]>([]);

  const addRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    const label = path.split("/").slice(-2).join("/");
    setRuns((r) => [...r, { path, label, entries }]);
  }, []);

  const removeRun = (path: string) => setRuns((r) => r.filter((x) => x.path !== path));

  const makeBarOption = (metricKey: string, metricLabel: string) => {
    const runLabels = runs.map((r) => r.label);
    const policies = [...new Set(runs.flatMap((r) => r.entries.map((e) => e.policy)))];

    const series = policies.map((p, i) => ({
      name: p,
      type: "bar",
      data: runs.map((r) => {
        const vals = r.entries
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[metricKey] ?? null)
          .filter((v): v is number => v !== null);
        return mean(vals);
      }),
      itemStyle: { color: COLORS[i % COLORS.length] },
    }));

    return {
      backgroundColor: "transparent",
      legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
      grid: { left: 50, right: 10, top: 35, bottom: 55 },
      xAxis: {
        type: "category",
        data: runLabels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: { type: "value", name: metricLabel, nameTextStyle: { color: "#9090b0" }, axisLabel: { color: "#9090b0", fontSize: 10 } },
      series,
      tooltip: { trigger: "axis" },
    };
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={addRun} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Add Run
        </button>
        <span className="text-xs text-canvas-muted">{runs.length} run(s) loaded</span>
      </div>

      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Loaded Runs
          </p>
          <div className="space-y-1">
            {runs.map((r) => (
              <div key={r.path} className="flex items-center gap-2 text-xs text-gray-300">
                <button
                  onClick={() => removeRun(r.path)}
                  className="text-canvas-muted hover:text-accent-danger"
                >
                  <X size={12} />
                </button>
                <span className="font-mono truncate">{r.label}</span>
                <span className="ml-auto text-canvas-muted">{r.entries.length} days</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {runs.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Add two or more run logs to compare them.
        </div>
      )}

      {runs.length >= 1 && (
        <div className="grid grid-cols-2 gap-4">
          {METRICS.map(({ key, label }) => (
            <div key={key} className="card">
              <p className="text-xs text-canvas-muted mb-2">{label}</p>
              <ReactECharts option={makeBarOption(key, label)} style={{ height: 220 }} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
