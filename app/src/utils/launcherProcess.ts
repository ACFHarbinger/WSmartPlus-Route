/**
 * Shared sim / data-gen / eval process detection for Studio launcher workflows (§D.7 / §G.9–§G.12).
 */
import type { ProcessEntry } from "../types";

export type LauncherKind = "sim" | "data_gen" | "eval";

export function isSimProcess(id: string, command: string): boolean {
  return id.startsWith("sim_") || /\btest_sim\b/.test(command);
}

export function isGenDataProcess(id: string, command: string): boolean {
  return id.startsWith("gen_data_") || /\bgen_data\b/.test(command);
}

export function isEvalProcess(id: string, command: string): boolean {
  return id.startsWith("eval_") || /\bmain\.py\s+eval\b/.test(command);
}

export function launcherKindFromProcess(
  id: string,
  command: string
): LauncherKind | null {
  if (isSimProcess(id, command)) return "sim";
  if (isGenDataProcess(id, command)) return "data_gen";
  if (isEvalProcess(id, command)) return "eval";
  return null;
}

/** Newest running process for a launcher kind, or null when none are active. */
export function findActiveLauncherProcessId(
  processes: Record<string, ProcessEntry>,
  kind: LauncherKind
): string | null {
  const running = Object.entries(processes)
    .filter(([id, proc]) => {
      if (proc.status !== "running") return false;
      const detected = launcherKindFromProcess(id, proc.command);
      return detected === kind;
    })
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return running[0]?.[0] ?? null;
}

function isRecentTerminalStatus(status: ProcessEntry["status"]): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

/** Newest launcher process that is running or recently finished (post-run panel persistence). */
export function findRecentLauncherProcessId(
  processes: Record<string, ProcessEntry>,
  kind: LauncherKind
): string | null {
  const candidates = Object.entries(processes)
    .filter(([id, proc]) => {
      if (proc.status !== "running" && !isRecentTerminalStatus(proc.status)) {
        return false;
      }
      return launcherKindFromProcess(id, proc.command) === kind;
    })
    .sort((a, b) => b[1].startTime - a[1].startTime);
  return candidates[0]?.[0] ?? null;
}

const EVAL_BATCH_WINDOW_MS = 30_000;

/** Recent eval processes from the same multi-checkpoint launch batch. */
export function findRecentEvalProcessIds(
  processes: Record<string, ProcessEntry>
): string[] {
  const candidates = Object.entries(processes).filter(([id, proc]) => {
    if (proc.status !== "running" && !isRecentTerminalStatus(proc.status)) {
      return false;
    }
    return isEvalProcess(id, proc.command);
  });
  if (candidates.length === 0) return [];

  const maxStart = Math.max(...candidates.map(([, proc]) => proc.startTime));
  return candidates
    .filter(([, proc]) => maxStart - proc.startTime <= EVAL_BATCH_WINDOW_MS)
    .sort((a, b) => a[1].startTime - b[1].startTime)
    .map(([id]) => id);
}

/** Shared live/post-run sim panel title for Simulation Launcher and Process Monitor (§G.9 / §G.15 / §D.7). */
export function simLivePanelTitle({
  isRunning,
  status,
}: {
  isRunning: boolean;
  status?: string;
}): string {
  if (isRunning) return "Live Status";

  const finalStatus = status ?? "completed";
  if (finalStatus === "completed") return "Run Complete";
  return `Run ${finalStatus}`;
}

/** Shared live/post-run data-gen panel title for Data Generation and Process Monitor (§G.11 / §G.15 / §D.7). */
export function dataGenLivePanelTitle({
  isRunning,
  status,
}: {
  isRunning: boolean;
  status?: string;
}): string {
  if (isRunning) return "Generating…";

  const finalStatus = status ?? "completed";
  if (finalStatus === "completed") return "Generation Complete";
  return `Generation ${finalStatus}`;
}
