/**
 * Simulation failure analysis parsing and display helpers (§A.6).
 *
 * Mirrors ``FailureAnalyzer`` stdout markers for the Studio Simulation Monitor.
 */
import type { SimFailureEntry, SimFailureSummary } from "../../types";

export const SIM_FAILURE_MARKER = "SIM_FAILURE_START:";

const CAUSE_LABELS: Record<string, string> = {
  overflow_event: "Overflow",
  waste_lost: "Waste Lost",
  negative_profit: "Negative Profit",
  fill_rate_spike: "Fill Rate Spike",
  skipped_high_fill: "Skipped High-Fill",
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: "#f87171",
  warning: "#fbbf24",
  info: "#38bdf8",
  ok: "#34d399",
};

export function parseSimFailureLine(line: string): SimFailureEntry | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(SIM_FAILURE_MARKER)) return null;
  const content = trimmed.slice(SIM_FAILURE_MARKER.length);
  const parts = content.split(",");
  if (parts.length < 4) return null;
  const policy = parts[0].trim();
  const sample_id = Number.parseInt(parts[1].trim(), 10);
  const day = Number.parseInt(parts[2].trim(), 10);
  const jsonPart = parts.slice(3).join(",").trim();
  try {
    const data = JSON.parse(jsonPart) as SimFailureSummary;
    if (!data.has_failure) return null;
    if (Number.isNaN(sample_id) || Number.isNaN(day)) return null;
    return { policy, sample_id, day, data };
  } catch {
    return null;
  }
}

export function simFailureCauseLabel(code: string): string {
  return CAUSE_LABELS[code] ?? code.replace(/_/g, " ");
}

export function simFailureSeverityColor(severity: string): string {
  return SEVERITY_COLORS[severity] ?? "#94a3b8";
}

export function filterFailureEntries(
  entries: SimFailureEntry[],
  policy: string | null,
  sampleId: number | null,
  day: number | null
): SimFailureEntry | null {
  return (
    entries.find(
      (e) =>
        (policy === null || e.policy === policy) &&
        (sampleId === null || e.sample_id === sampleId) &&
        (day === null || e.day === day)
    ) ?? null
  );
}
