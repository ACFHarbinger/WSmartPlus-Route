/**
 * Training health alert parsing and display helpers (§A.4).
 *
 * Mirrors ``TrainingHealthCallback`` stdout markers for the Studio Training Monitor.
 */
import type { TrainingHealthEntry } from "../../types";

export const TRAINING_HEALTH_MARKER = "TRAINING_HEALTH_START:";

export type TrainingHealthCode =
  | "grad_norm_explosion"
  | "reward_stagnation"
  | "entropy_collapse"
  | string;

const CODE_LABELS: Record<string, string> = {
  grad_norm_explosion: "Gradient Explosion",
  reward_stagnation: "Reward Stagnation",
  entropy_collapse: "Entropy Collapse",
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "#f87171",
  warning: "#fbbf24",
};

export function parseTrainingHealthLine(line: string): TrainingHealthEntry | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(TRAINING_HEALTH_MARKER)) return null;
  const jsonPart = trimmed.slice(TRAINING_HEALTH_MARKER.length);
  try {
    const payload = JSON.parse(jsonPart) as TrainingHealthEntry;
    if (!payload.code || !payload.severity) return null;
    return payload;
  } catch {
    return null;
  }
}

export function trainingHealthCodeLabel(code: TrainingHealthCode): string {
  return CODE_LABELS[code] ?? code.replace(/_/g, " ");
}

export function trainingHealthSeverityColor(severity: string): string {
  return SEVERITY_COLORS[severity] ?? "#94a3b8";
}

export function countByCode(entries: TrainingHealthEntry[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const e of entries) {
    counts[e.code] = (counts[e.code] ?? 0) + 1;
  }
  return counts;
}

export function sortByEpochStep(entries: TrainingHealthEntry[]): TrainingHealthEntry[] {
  return [...entries].sort((a, b) => {
    if (a.epoch !== b.epoch) return a.epoch - b.epoch;
    return a.step - b.step;
  });
}

/** Parse all ``TRAINING_HEALTH_START:`` markers from process stdout (§A.4). */
export function collectTrainingHealthFromLogLines(lines: string[]): TrainingHealthEntry[] {
  const entries: TrainingHealthEntry[] = [];
  for (const line of lines) {
    const parsed = parseTrainingHealthLine(line);
    if (parsed) entries.push(parsed);
  }
  return entries;
}
