/**
 * Simulation Summary — post-run aggregate analytics.
 * Ports Streamlit `simulation_summary` mode.
 */
import { useCallback, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen } from "lucide-react";
import { KpiCard } from "../components/ui/KpiCard";
import type { DayLogEntry } from "../types";

function aggregateByPolicy(entries: DayLogEntry[]) {
  const map: Record<string, { profit: number[]; km: number[]; overflows: number[]; kg: number[] }> = {};
  for (const e of entries) {
    if (!map[e.policy]) map[e.policy] = { profit: [], km: [], overflows: [], kg: [] };
    const d = e.data as Record<string, number>;
    if (d.profit != null) map[e.policy].profit.push(d.profit);
    if (d.km != null) map[e.policy].km.push(d.km);
    if (d.overflows != null) map[e.policy].overflows.push(d.overflows);
    if (d.kg != null) map[e.policy].kg.push(d.kg);
  }
  return map;
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function SimulationSummary() {
  const [entries, setEntries] = useState<DayLogEntry[]>([]);

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    setEntries(loaded);
  }, []);

  const agg = aggregateByPolicy(entries);
  const policies = Object.keys(agg);

  const profitOption = {
    backgroundColor: "transparent",
    xAxis: { type: "category", data: policies, axisLabel: { color: "#9090b0", fontSize: 10 } },
    yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 10 } },
    grid: { left: 50, right: 10, top: 20, bottom: 50 },
    series: [
      {
        type: "bar",
        data: policies.map((p) => mean(agg[p].profit)),
        itemStyle: { color: "#6366f1" },
        name: "Avg Profit (€)",
      },
    ],
    tooltip: { trigger: "axis" },
  };

  const overflowOption = {
    backgroundColor: "transparent",
    xAxis: { type: "category", data: policies, axisLabel: { color: "#9090b0", fontSize: 10 } },
    yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 10 } },
    grid: { left: 50, right: 10, top: 20, bottom: 50 },
    series: [
      {
        type: "bar",
        data: policies.map((p) => mean(agg[p].overflows)),
        itemStyle: { color: "#f87171" },
        name: "Avg Overflows",
      },
    ],
    tooltip: { trigger: "axis" },
  };

  return (
    <div className="space-y-4">
      <button onClick={openLog} className="btn-primary flex items-center gap-2">
        <FolderOpen size={14} />
        Open Log File
      </button>

      {entries.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Load a completed simulation log to view summary analytics.
        </div>
      )}

      {policies.length > 0 && (
        <>
          {/* Per-policy average KPIs */}
          {policies.map((p) => (
            <div key={p} className="card">
              <p className="text-sm font-semibold text-gray-200 mb-3">{p}</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <KpiCard label="Avg Profit (€)" value={mean(agg[p].profit)} lowerIsBetter={false} />
                <KpiCard label="Avg Distance (km)" value={mean(agg[p].km)} lowerIsBetter={true} />
                <KpiCard label="Avg Overflows" value={mean(agg[p].overflows)} lowerIsBetter={true} />
                <KpiCard label="Avg Waste (kg)" value={mean(agg[p].kg)} lowerIsBetter={false} />
              </div>
            </div>
          ))}

          {/* Policy comparison charts */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card">
              <p className="text-xs text-canvas-muted mb-2">Avg Profit by Policy</p>
              <ReactECharts option={profitOption} style={{ height: 200 }} />
            </div>
            <div className="card">
              <p className="text-xs text-canvas-muted mb-2">Avg Overflows by Policy</p>
              <ReactECharts option={overflowOption} style={{ height: 200 }} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
