/**
 * Shared eval stdout JSON parsing for Evaluation Runner and Process Monitor (§G.12 / §G.15).
 */
import type { EvalAnalyticsRow } from "../types";

export const EVAL_RESULT_KEYS = [
  "cost",
  "gap",
  "tour_cost",
  "obj",
  "time",
  "policy",
  "checkpoint",
] as const;

export interface EvalResult {
  checkpointName: string;
  cost?: number;
  gap?: number;
  time?: number;
  policy?: string;
  [key: string]: number | string | undefined;
}

/** Extract checkpoint load path from a Hydra eval command line, if present. */
export function checkpointPathFromEvalCommand(command: string): string | null {
  const match = command.match(/eval\.policy\.model\.load_path=([^\s]+)/);
  if (!match) return null;
  return match[1].replace(/^['"]|['"]$/g, "");
}

/** Derive a human-readable checkpoint label from process id or Hydra command. */
export function checkpointLabelFromEvalProcess(id: string, command: string): string {
  if (id.startsWith("eval_")) {
    const parts = id.slice(5).split("_");
    if (parts.length >= 3) {
      return parts.slice(0, -2).join("_");
    }
    return parts[0] ?? id;
  }

  const match = command.match(/eval\.policy\.model\.load_path=([^\s]+)/);
  if (match) {
    const path = match[1].replace(/^['"]|['"]$/g, "");
    return path.split(/[/\\]/).pop() ?? id;
  }

  return id;
}

export function parseEvalResultLine(line: string): Partial<EvalResult> | null {
  const text = line.startsWith("[stderr]") ? line.slice(8) : line;
  try {
    const obj = JSON.parse(text) as Record<string, unknown>;
    if (!EVAL_RESULT_KEYS.some((k) => obj[k] != null)) return null;

    const result: Partial<EvalResult> = {};
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === "number" || typeof v === "string") {
        result[k] = v;
      }
    }
    return result;
  } catch {
    return null;
  }
}

/** Merge all structured eval JSON lines from a process log into one result row. */
export function collectEvalResultFromLogLines(
  logLines: string[],
  checkpointName: string
): EvalResult {
  const result: EvalResult = { checkpointName };

  for (const line of logLines) {
    const partial = parseEvalResultLine(line);
    if (!partial) continue;

    Object.assign(result, partial);
    if (typeof partial.checkpoint === "string" && partial.checkpoint.trim() !== "") {
      result.checkpointName = partial.checkpoint;
    }
  }

  return result;
}

export function hasEvalMetrics(result: EvalResult): boolean {
  return EVAL_RESULT_KEYS.some((k) => {
    if (k === "checkpoint") return false;
    return result[k] != null;
  });
}

/** Shared live/post-run eval panel title for Training Hub, Evaluation Runner, and Process Monitor (§G.10 / §G.12 / §G.15 / §D.7). */
export function evalLivePanelTitle({
  isRunning,
  status,
  total = 1,
  runningCount = 0,
  completedCount = 0,
}: {
  isRunning: boolean;
  status?: string;
  total?: number;
  runningCount?: number;
  completedCount?: number;
}): string {
  if (isRunning) {
    if (total > 1) {
      return `Evaluating ${total - runningCount}/${total}…`;
    }
    return "Evaluating…";
  }

  const finalStatus = status ?? "completed";
  if (finalStatus === "completed") {
    if (total > 1 && completedCount > 1) {
      return `Evaluation Complete (${completedCount}/${total})`;
    }
    return "Evaluation Complete";
  }

  return `Evaluation ${finalStatus}`;
}

export function toEvalAnalyticsRows(results: EvalResult[]): EvalAnalyticsRow[] {
  return results.map((r) => ({
    checkpoint: r.checkpointName,
    cost: r.cost,
    gap: r.gap,
    time: r.time,
    policy: r.policy,
  }));
}
