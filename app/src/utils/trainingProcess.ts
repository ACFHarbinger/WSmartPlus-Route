/**
 * Shared train/HPO process detection for Studio live analytics (§A.4 / §A.2).
 */
import type { ProcessEntry } from "../types";

export function isTrainOrHpoProcess(id: string, command: string): boolean {
  return (
    id.startsWith("train_") ||
    id.startsWith("hpo_") ||
    id.startsWith("meta_") ||
    /\bmain\.py\s+(train|hpo|meta_train)\b/.test(command)
  );
}

export function isMetaTrainProcess(id: string, command: string): boolean {
  return id.startsWith("meta_") || /\bmain\.py\s+meta_train\b/.test(command);
}

export function isHpoProcess(id: string, command: string): boolean {
  return id.startsWith("hpo_") || /\bmain\.py\s+hpo\b/.test(command);
}

export function isTrainProcess(id: string, command: string): boolean {
  return (
    id.startsWith("train_") ||
    (/\bmain\.py\s+train\b/.test(command) && !isHpoProcess(id, command))
  );
}

/** Newest meta-RL training process that is running or recently finished (Training Hub meta mode). */
export function findRecentMetaTrainProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        (proc.status === "running" || isRecentTerminalStatus(proc.status)) &&
        isMetaTrainProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
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

function isRecentTerminalStatus(status: ProcessEntry["status"]): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

/** Newest train/HPO process that is running or recently finished (for post-run deep-links). */
export function findRecentTrainOrHpoProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        (proc.status === "running" || isRecentTerminalStatus(proc.status)) &&
        isTrainOrHpoProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

/** Newest HPO process that is running or recently finished (HPO Tracker / Experiment Tracker). */
export function findRecentHpoProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        (proc.status === "running" || isRecentTerminalStatus(proc.status)) &&
        isHpoProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

/** Newest train-only process that is running or recently finished (Training Hub train mode). */
export function findRecentTrainProcessId(
  processes: Record<string, ProcessEntry>
): string | null {
  const candidates = Object.entries(processes)
    .filter(
      ([id, proc]) =>
        (proc.status === "running" || isRecentTerminalStatus(proc.status)) &&
        isTrainProcess(id, proc.command)
    )
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

export function liveTrainProcessLabel(id: string): string {
  return trainHpoLivePanelTitle({ isRunning: true, processId: id });
}

/** Shared live/post-run train/HPO panel title for Training Hub, monitors, and trackers (§G.10 / §G.15 / §G.17 / §G.18 / §D.7). */
export function trainHpoLivePanelTitle({
  isRunning,
  status,
  processId,
  command,
  kind,
}: {
  isRunning: boolean;
  status?: string;
  processId?: string;
  command?: string;
  /** Explicit override when process id is not yet known (e.g. Training Hub mode selector). */
  kind?: "train" | "hpo";
}): string {
  const resolvedKind =
    kind ??
    (processId
      ? isHpoProcess(processId, command ?? "")
        ? "hpo"
        : "train"
      : "train");

  if (isRunning) {
    return resolvedKind === "hpo" ? "Live HPO" : "Live Training";
  }

  const finalStatus = status ?? "completed";
  if (finalStatus === "completed") {
    return resolvedKind === "hpo" ? "HPO Complete" : "Training Complete";
  }

  return resolvedKind === "hpo"
    ? `HPO ${finalStatus}`
    : `Training ${finalStatus}`;
}
