/**
 * Epoch/trial progress bar + ETA for live train/HPO processes (§D.2 / §G.10 / §G.17).
 */
import { useEffect, useMemo, useState } from "react";
import { useProcessStore } from "../../store/process";
import {
  computeEtaMs,
  formatDurationMs,
  getLatestProgress,
  progressPercent,
} from "../../utils/processProgress";

export interface LiveTrainProgressBarProps {
  processId: string;
  /** Fallback epoch total from the Training Hub form when PROGRESS markers omit total. */
  fallbackTotal?: number;
  /** Fallback current epoch from the latest parsed metric row. */
  fallbackValue?: number;
  className?: string;
}

export function LiveTrainProgressBar({
  processId,
  fallbackTotal,
  fallbackValue,
  className = "",
}: LiveTrainProgressBarProps) {
  const proc = useProcessStore((s) => s.processes[processId]);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!proc || proc.status !== "running") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [proc]);

  const progress = useMemo(() => {
    if (!proc || proc.status !== "running") return null;

    const marker = getLatestProgress(proc.logLines);
    const value = marker?.value ?? fallbackValue;
    const total = marker?.total ?? fallbackTotal;
    const label = marker?.label;

    if (value == null) return null;

    const info = { value, total, label };
    const pct = progressPercent(info);
    const elapsedMs = now - proc.startTime;
    const etaMs =
      total != null && total > 0 ? computeEtaMs(value, total, elapsedMs) : null;

    return {
      info,
      pct,
      elapsedLabel: formatDurationMs(elapsedMs),
      etaLabel: etaMs != null ? formatDurationMs(etaMs) : null,
    };
  }, [proc, fallbackTotal, fallbackValue, now]);

  if (!progress) return null;

  const { info, pct, elapsedLabel, etaLabel } = progress;

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="flex-1 h-1.5 bg-canvas-elevated rounded-full overflow-hidden">
        {pct != null ? (
          <div
            className="h-full bg-accent-primary rounded-full transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full bg-accent-primary/40 rounded-full animate-pulse w-full" />
        )}
      </div>
      <span className="text-[10px] font-mono text-canvas-muted shrink-0">
        {pct != null ? `${pct.toFixed(0)}%` : info.label ?? `${info.value}`}
      </span>
      {info.total != null && (
        <span className="text-[10px] font-mono text-canvas-muted shrink-0">
          {info.value}/{info.total}
        </span>
      )}
      {info.label && pct != null && (
        <span className="text-[10px] text-canvas-muted shrink-0 hidden sm:inline">
          {info.label}
        </span>
      )}
      <span className="text-[10px] text-canvas-muted shrink-0" title="Elapsed">
        {elapsedLabel}
      </span>
      {etaLabel && (
        <span className="text-[10px] text-accent-secondary shrink-0" title="Estimated remaining">
          ETA {etaLabel}
        </span>
      )}
    </div>
  );
}
