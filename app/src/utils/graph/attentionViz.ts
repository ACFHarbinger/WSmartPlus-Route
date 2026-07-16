/**
 * Runtime attention ring-buffer parsing and chart helpers (§A.2 Option A).
 */
import { buildMatrixHeatmapOption } from "../charts/tensorHeatmap";
import { transformMatrixLogScale } from "../charts/chartLogScale";
import type { AttentionSnapshot, AttentionVizEntry } from "../../types";

export const ATTENTION_VIZ_MARKER = "ATTENTION_VIZ_START:";

export function parseAttentionVizLine(line: string): AttentionVizEntry | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(ATTENTION_VIZ_MARKER)) return null;
  const jsonPart = trimmed.slice(ATTENTION_VIZ_MARKER.length);
  try {
    const payload = JSON.parse(jsonPart) as AttentionVizEntry;
    if (!payload.phase || !payload.snapshots?.length) return null;
    return payload;
  } catch {
    return null;
  }
}

export function sortAttentionEntries(entries: AttentionVizEntry[]): AttentionVizEntry[] {
  return [...entries].sort((a, b) => {
    if (a.epoch !== b.epoch) return a.epoch - b.epoch;
    return a.step - b.step;
  });
}

export function latestAttentionEntry(entries: AttentionVizEntry[]): AttentionVizEntry | null {
  const sorted = sortAttentionEntries(entries);
  return sorted.length ? sorted[sorted.length - 1]! : null;
}

export function listSnapshotOptions(entry: AttentionVizEntry): AttentionSnapshot[] {
  return [...entry.snapshots].sort((a, b) => {
    if (a.decode_step !== b.decode_step) return a.decode_step - b.decode_step;
    if (a.layer !== b.layer) return a.layer - b.layer;
    return a.head - b.head;
  });
}

export function snapshotKey(snap: AttentionSnapshot): string {
  return `d${snap.decode_step}-L${snap.layer}-H${snap.head}`;
}

/** Parse all ``ATTENTION_VIZ_START:`` markers from process stdout (§A.2 Option A). */
export function collectAttentionVizFromLogLines(lines: string[]): AttentionVizEntry[] {
  const entries: AttentionVizEntry[] = [];
  for (const line of lines) {
    const parsed = parseAttentionVizLine(line);
    if (parsed) entries.push(parsed);
  }
  return entries;
}

export function buildRuntimeAttentionHeatmapOption(
  snap: AttentionSnapshot,
  theme: "dark" | "light",
  logScale = false
): Record<string, unknown> {
  const rawValues = snap.matrix;
  const displayValues = logScale
    ? transformMatrixLogScale(rawValues, "attention", true)
    : rawValues;
  const title = `Runtime attention · layer ${snap.layer} · head ${snap.head} · step ${snap.decode_step}${
    logScale ? " · log colour" : ""
  }`;
  const base = buildMatrixHeatmapOption(displayValues, {
    title,
    theme,
    xLabel: "Key",
    yLabel: "Query",
  });
  if (!logScale) return base;
  return {
    ...base,
    tooltip: {
      position: "top",
      formatter: (p: { value?: [number, number, number] }) => {
        const v = p.value;
        if (!v) return "";
        const raw = rawValues[v[1]]?.[v[0]];
        const rawText = Number.isFinite(raw) ? Number(raw).toFixed(4) : "—";
        return `Query ${v[1]} · Key ${v[0]}<br/>${rawText}`;
      },
    },
  };
}
