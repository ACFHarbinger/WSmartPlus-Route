/**
 * Training health alert dashboard (§A.4).
 *
 * Renders instability warnings from ``TrainingHealthCallback`` streamed via
 * ``TRAINING_HEALTH_START:`` log markers.
 */
import { useMemo } from "react";
import { AlertTriangle, ShieldAlert } from "lucide-react";
import {
  countByCode,
  sortByEpochStep,
  trainingHealthCodeLabel,
  trainingHealthSeverityColor,
} from "../../utils/trainingHealth";
import type { TrainingHealthEntry } from "../../types";

interface Props {
  entries: TrainingHealthEntry[];
}

export function TrainingHealthPanel({ entries }: Props) {
  const sorted = useMemo(() => sortByEpochStep(entries), [entries]);
  const counts = useMemo(() => countByCode(entries), [entries]);

  if (sorted.length === 0) {
    return (
      <div className="card p-4 text-xs text-canvas-muted">
        <p className="flex items-center gap-2 font-medium text-gray-400">
          <ShieldAlert size={14} />
          Training Health
        </p>
        <p className="mt-2">
          No instability alerts yet. The health callback monitors gradient norm,
          reward stagnation, and entropy collapse during Lightning training runs.
        </p>
      </div>
    );
  }

  const criticalCount = sorted.filter((e) => e.severity === "critical").length;
  const warningCount = sorted.filter((e) => e.severity === "warning").length;

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="flex items-center gap-2 text-sm font-semibold text-gray-200">
            <ShieldAlert size={14} className="text-accent-primary" />
            Training Health
            <span className="text-xs font-normal text-canvas-muted">
              {sorted.length} alert{sorted.length !== 1 ? "s" : ""}
              {criticalCount > 0 && (
                <span className="text-red-400 ml-1">· {criticalCount} critical</span>
              )}
              {warningCount > 0 && (
                <span className="text-amber-400 ml-1">· {warningCount} warning</span>
              )}
            </span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(counts).map(([code, n]) => (
            <span
              key={code}
              className="text-[10px] px-2 py-0.5 rounded-full border border-canvas-border text-canvas-muted"
            >
              {trainingHealthCodeLabel(code)} × {n}
            </span>
          ))}
        </div>
      </div>

      <div className="max-h-64 overflow-y-auto divide-y divide-canvas-border/40 rounded-lg border border-canvas-border">
        {[...sorted].reverse().map((entry, idx) => {
          const color = trainingHealthSeverityColor(entry.severity);
          return (
            <div key={`${entry.code}-${entry.epoch}-${entry.step}-${idx}`} className="px-3 py-2.5">
              <div className="flex items-start gap-2">
                <AlertTriangle size={13} style={{ color }} className="shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className="text-xs font-semibold uppercase tracking-wide"
                      style={{ color }}
                    >
                      {entry.severity}
                    </span>
                    <span className="text-xs text-gray-300">
                      {trainingHealthCodeLabel(entry.code)}
                    </span>
                    <span className="text-[10px] text-canvas-muted font-mono">
                      epoch {entry.epoch + 1} · step {entry.step}
                    </span>
                  </div>
                  <p className="text-xs text-canvas-muted mt-0.5">{entry.message}</p>
                  {entry.details && Object.keys(entry.details).length > 0 && (
                    <p className="text-[10px] text-canvas-muted font-mono mt-1 truncate">
                      {JSON.stringify(entry.details)}
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
