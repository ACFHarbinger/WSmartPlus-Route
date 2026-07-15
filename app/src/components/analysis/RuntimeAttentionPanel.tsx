/**
 * Runtime encoder attention heatmaps from ``AttentionRingBuffer`` (§A.2 Option A).
 *
 * Renders snapshots streamed via ``ATTENTION_VIZ_START:`` markers during
 * eval/validation when ``tracking.log_attention`` is enabled.
 */
import { useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Brain } from "lucide-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import {
  buildRuntimeAttentionHeatmapOption,
  latestAttentionEntry,
  listSnapshotOptions,
  snapshotKey,
  sortAttentionEntries,
} from "../../utils/attentionViz";
import type { AttentionSnapshot, AttentionVizEntry } from "../../types";

interface Props {
  entries: AttentionVizEntry[];
  theme: "dark" | "light";
  logScale?: boolean;
}

export function RuntimeAttentionPanel({ entries, theme, logScale = false }: Props) {
  const chartRef = useRef<EChartsReact>(null);
  const sorted = useMemo(() => sortAttentionEntries(entries), [entries]);
  const defaultEntry = useMemo(() => latestAttentionEntry(entries), [entries]);
  const [selectedEntry, setSelectedEntry] = useState<AttentionVizEntry | null>(null);
  const activeEntry = selectedEntry ?? defaultEntry;
  const snapshots = useMemo(
    () => (activeEntry ? listSnapshotOptions(activeEntry) : []),
    [activeEntry]
  );

  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const activeKey = selectedKey ?? (snapshots[0] ? snapshotKey(snapshots[0]) : null);

  const activeSnapshot: AttentionSnapshot | null = useMemo(() => {
    if (!activeEntry || !activeKey) return null;
    return snapshots.find((s) => snapshotKey(s) === activeKey) ?? snapshots[0] ?? null;
  }, [activeEntry, activeKey, snapshots]);

  const heatmapOption = useMemo(
    () =>
      activeSnapshot
        ? buildRuntimeAttentionHeatmapOption(activeSnapshot, theme, logScale)
        : null,
    [activeSnapshot, theme, logScale]
  );

  if (!activeEntry || !activeSnapshot || !heatmapOption) {
    return (
      <div className="card p-4 text-xs text-canvas-muted">
        <p className="flex items-center gap-2 font-medium text-gray-400">
          <Brain size={14} />
          Runtime Attention
        </p>
        <p className="mt-2">
          No runtime attention captures yet. Enable ``tracking.log_attention`` during
          training or evaluation to stream encoder heatmaps to the Studio.
        </p>
      </div>
    );
  }

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="flex items-center gap-2 text-sm font-semibold text-gray-200">
            <Brain size={14} className="text-accent-primary" />
            Runtime Attention
            <span className="text-xs font-normal text-canvas-muted">
              {activeEntry.phase} · epoch {activeEntry.epoch} · step {activeEntry.step} ·{" "}
              {snapshots.length} snapshot{snapshots.length !== 1 ? "s" : ""}
            </span>
          </p>
          <p className="text-[10px] text-canvas-muted mt-0.5">
            Ring-buffer captures from encoder forward hooks · {sorted.length} emission
            {sorted.length !== 1 ? "s" : ""} loaded
          </p>
        </div>
        <ChartExportButtons chartRef={chartRef} basename="runtime-attention" />
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-canvas-muted">
          Snapshot
          <select
            className="select-base text-xs py-0.5 max-w-[220px]"
            value={activeKey ?? ""}
            onChange={(e) => setSelectedKey(e.target.value)}
          >
            {snapshots.map((snap) => {
              const key = snapshotKey(snap);
              return (
                <option key={key} value={key}>
                  {key} · {snap.n_nodes} nodes
                </option>
              );
            })}
          </select>
        </label>
        {sorted.length > 1 && (
          <label className="flex items-center gap-1.5 text-canvas-muted">
            Capture
            <select
              className="select-base text-xs py-0.5"
              value={`${activeEntry.epoch}-${activeEntry.step}`}
              onChange={(e) => {
                const [epoch, step] = e.target.value.split("-").map(Number);
                const match = sorted.find((entry) => entry.epoch === epoch && entry.step === step);
                if (match) {
                  setSelectedEntry(match);
                  if (match.snapshots[0]) {
                    setSelectedKey(snapshotKey(match.snapshots[0]));
                  }
                }
              }}
            >
              {[...sorted].reverse().map((entry) => (
                <option key={`${entry.epoch}-${entry.step}`} value={`${entry.epoch}-${entry.step}`}>
                  {entry.phase} epoch {entry.epoch} step {entry.step}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <ReactECharts
        ref={chartRef}
        option={heatmapOption}
        style={{ height: 360 }}
        theme={theme === "dark" ? "dark" : undefined}
        notMerge
      />
    </div>
  );
}
