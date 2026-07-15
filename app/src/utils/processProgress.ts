/** PROGRESS:{json} markers emitted by Python subprocesses (§G.15 / §D.2). */

export interface ProgressInfo {
  value: number;
  total?: number;
  label?: string;
}

const PROGRESS_MARKER = "PROGRESS:";

/** Scan recent stdout lines for the latest structured progress marker. */
export function getLatestProgress(logLines: string[]): ProgressInfo | null {
  const start = Math.max(0, logLines.length - 30);
  for (let i = logLines.length - 1; i >= start; i--) {
    const line = logLines[i];
    const idx = line.indexOf(PROGRESS_MARKER);
    if (idx === -1) continue;
    try {
      const raw = JSON.parse(line.slice(idx + PROGRESS_MARKER.length).trim()) as Record<
        string,
        unknown
      >;
      const value =
        typeof raw.value === "number"
          ? raw.value
          : typeof raw.current === "number"
            ? raw.current
            : null;
      if (value === null) continue;
      return {
        value,
        total: typeof raw.total === "number" ? raw.total : undefined,
        label: typeof raw.label === "string" ? raw.label : undefined,
      };
    } catch {
      /* ignore malformed markers */
    }
  }
  return null;
}

export function progressPercent(prog: ProgressInfo): number | null {
  if (prog.total == null || prog.total <= 0) return null;
  return Math.min(100, (prog.value / prog.total) * 100);
}

/** Human-readable duration from milliseconds. */
export function formatDurationMs(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/**
 * Estimate remaining time from deterministic progress (epoch/trial counters).
 * Returns null when progress is unknown or already complete.
 */
export function computeEtaMs(value: number, total: number, elapsedMs: number): number | null {
  if (value <= 0 || value >= total || elapsedMs <= 0) return null;
  const msPerUnit = elapsedMs / value;
  return msPerUnit * (total - value);
}
