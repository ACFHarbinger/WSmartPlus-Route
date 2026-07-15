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
