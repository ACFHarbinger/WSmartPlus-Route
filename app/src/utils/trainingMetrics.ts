/**
 * Training metric stdout parsing for launcher live panels (§G.10 / §G.17).
 */
import type { TrainingMetricsRow } from "../types";

const METRIC_SIGNAL_KEYS = [
  "train_loss", "train/rl_loss", "train/il_loss",
  "val_loss", "val/cost", "val_cost",
  "reward", "grad_norm", "entropy", "epoch", "step",
];

export function normalizeTrainingMetricRow(raw: Record<string, unknown>): TrainingMetricsRow {
  const r = { ...raw } as TrainingMetricsRow;
  if (r.train_loss == null) {
    r.train_loss = (raw["train/rl_loss"] ?? raw["train/il_loss"]) as number | undefined;
  }
  if (r.val_loss == null) {
    r.val_loss = (raw["val/cost"] ?? raw["val_cost"]) as number | undefined;
  }
  if (r.lr == null) {
    for (const key of Object.keys(raw)) {
      if (key !== "lr" && key.startsWith("lr") && typeof raw[key] === "number") {
        r.lr = raw[key] as number;
        break;
      }
    }
  }
  return r;
}

export function parseTrainingMetricLine(line: string): TrainingMetricsRow | null {
  const text = line.startsWith("[stderr]") ? line.slice(8) : line;
  try {
    const obj = JSON.parse(text) as Record<string, unknown>;
    if (METRIC_SIGNAL_KEYS.some((k) => typeof obj[k] === "number")) {
      return normalizeTrainingMetricRow(obj);
    }
  } catch {
    // fall through to key=value parsing
  }
  if (METRIC_SIGNAL_KEYS.some((k) => text.includes(k))) {
    const row: Record<string, number> = {};
    for (const [, key, val] of text.matchAll(/(\w[\w/]*)=([0-9.eE+\-]+)/g)) {
      row[key] = parseFloat(val);
    }
    if (Object.keys(row).length > 0) return normalizeTrainingMetricRow(row);
  }
  return null;
}

/** Parse all training metric rows from process stdout. */
export function collectTrainingMetricsFromLogLines(
  lines: string[]
): TrainingMetricsRow[] {
  const rows: TrainingMetricsRow[] = [];
  for (const line of lines) {
    const row = parseTrainingMetricLine(line);
    if (row) rows.push(row);
  }
  return rows;
}

/** Post-run banner text when train/HPO panels rehydrate from ``useProcessStore``. */
export function postRunTrainingRehydrationMessage({
  metricCount,
  healthCount = 0,
  attentionCount = 0,
  fallback = "Post-run shortcuts — open Training Monitor or Output Browser for this run",
}: {
  metricCount: number;
  healthCount?: number;
  attentionCount?: number;
  fallback?: string;
}): string {
  if (metricCount === 0 && healthCount === 0 && attentionCount === 0) {
    return fallback;
  }
  const parts: string[] = [];
  if (metricCount > 0) parts.push("metrics + sparklines");
  if (healthCount > 0) parts.push("health alerts");
  if (attentionCount > 0) parts.push("attention snapshots");
  return `Post-run ${parts.join(", ")} rehydrated from process store — persist after navigation`;
}
