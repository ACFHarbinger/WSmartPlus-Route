/**
 * Derive assets/output run directories from process stdout (§G.14 / §G.9 / §G.15).
 */
import { extractJsonlPathFromLogLines } from "./policyTelemetryTrends";

/** Resolve the Hydra run root from a simulation ``.jsonl`` log path. */
export function outputRunPathFromJsonl(jsonlPath: string): string {
  const normalized = jsonlPath.replace(/\\/g, "/");
  const hydraIdx = normalized.lastIndexOf("/hydra/");
  if (hydraIdx !== -1) {
    return normalized.slice(0, hydraIdx);
  }
  const lastSlash = normalized.lastIndexOf("/");
  if (lastSlash === -1) {
    return normalized;
  }
  return normalized.slice(0, lastSlash);
}

/** Scan process stdout for the newest ``.jsonl`` path and return its run directory. */
export function outputRunPathFromLogLines(lines: string[]): string | null {
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (!jsonl) return null;
  return outputRunPathFromJsonl(jsonl);
}
