/**
 * Training metric stdout parsing for launcher live panels (§G.10 / §G.17).
 */
import type { TrainingMetricsRow } from "../types";

const METRIC_SIGNAL_KEYS = [
  "train_loss", "train/rl_loss", "train/il_loss",
  "val_loss", "val/cost", "val_cost",
  "reward", "grad_norm", "entropy", "epoch", "step",
];

function normalizeMetricRow(raw: Record<string, unknown>): TrainingMetricsRow {
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
      return normalizeMetricRow(obj);
    }
  } catch {
    // fall through to key=value parsing
  }
  if (METRIC_SIGNAL_KEYS.some((k) => text.includes(k))) {
    const row: Record<string, number> = {};
    for (const [, key, val] of text.matchAll(/(\w[\w/]*)=([0-9.eE+\-]+)/g)) {
      row[key] = parseFloat(val);
    }
    if (Object.keys(row).length > 0) return normalizeMetricRow(row);
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
