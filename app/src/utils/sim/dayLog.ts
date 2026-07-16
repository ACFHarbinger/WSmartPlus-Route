/**
 * Simulation day-log marker parsing for launcher live panels (§G.9).
 */
import type { DayLogEntry } from "../../types";

export const GUI_DAY_LOG_MARKER = "GUI_DAY_LOG_START:";

export function parseDayLogLine(line: string): DayLogEntry | null {
  const markerIdx = line.indexOf(GUI_DAY_LOG_MARKER);
  if (markerIdx === -1) return null;
  const jsonStr = line.slice(markerIdx + GUI_DAY_LOG_MARKER.length).trim();
  try {
    const entry = JSON.parse(jsonStr) as DayLogEntry;
    if (!entry.policy) return null;
    return entry;
  } catch {
    return null;
  }
}

/** Latest day-log entry per policy/sample from process stdout (§G.9). */
export function collectLatestDayLogsByPolicy(
  lines: string[]
): Record<string, DayLogEntry> {
  const latest: Record<string, DayLogEntry> = {};
  for (const line of lines) {
    const entry = parseDayLogLine(line);
    if (!entry) continue;
    latest[`${entry.policy}::${entry.sample_id}`] = entry;
  }
  return latest;
}
