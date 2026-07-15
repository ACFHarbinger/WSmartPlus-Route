/**
 * Shared train/HPO process detection for Studio live analytics (§A.4 / §A.2).
 */
import type { ProcessEntry } from "../types";

export function isTrainOrHpoProcess(id: string, command: string): boolean {
  return (
    id.startsWith("train_") ||
    id.startsWith("hpo_") ||
    /\bmain\.py\s+(train|hpo)\b/.test(command)
  );
}

export function isHpoProcess(id: string, command: string): boolean {
  return id.startsWith("hpo_") || /\bmain\.py\s+hpo\b/.test(command);
}

/** Newest running train/HPO process, or null when none are active. */
export function findActiveLiveTrainProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const running = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        proc.status === "running" && isTrainOrHpoProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return running[0]?.[0] ?? null;
}

/** Newest running HPO process only (for HPO Tracker live panels). */
export function findActiveHpoProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const running = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        proc.status === "running" && isHpoProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return running[0]?.[0] ?? null;
}

export function liveTrainProcessLabel(id: string): string {
  return id.startsWith("hpo_") ? "Live HPO" : "Live Training";
}
