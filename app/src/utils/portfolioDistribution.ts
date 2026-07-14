/**
 * Group portfolio runs by distribution label for facet heatmaps (§G.1.3).
 */

import { parseLogPath } from "./simMetadata";
import type { DayLogEntry } from "../types";

export interface PortfolioRunRef {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

/** Bucket loaded runs by parsed distribution (Empirical, Gamma-3, …). */
export function groupRunsByDistribution(
  runs: PortfolioRunRef[]
): Array<[string, PortfolioRunRef[]]> {
  const map = new Map<string, PortfolioRunRef[]>();
  for (const run of runs) {
    const dist = parseLogPath(run.path).distribution ?? "Unknown";
    const list = map.get(dist) ?? [];
    list.push(run);
    map.set(dist, list);
  }
  return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
}
