/**
 * Build per-panel Pareto scatter points from loaded simulation runs (§G.1.2).
 */

import { parseLogPath } from "./simMetadata";
import { panelForRun, PARETO_PANELS } from "./paretoPanels";
import type { DayLogEntry } from "../types";

export interface PortfolioRunSlice {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

export interface ParetoPoint {
  id: string;
  x: number;
  y: number;
  policy: string;
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

/** Classify runs into the four Gamma/Empirical × FTSP/CLS panels. */
export function buildParetoByPanel(
  runs: PortfolioRunSlice[]
): Record<string, ParetoPoint[]> {
  const panels: Record<string, ParetoPoint[]> = {};
  for (const panel of PARETO_PANELS) panels[panel.id] = [];

  for (const run of runs) {
    const panelId = panelForRun(parseLogPath(run.path));
    if (!panelId) continue;
    const policies = [...new Set(run.entries.map((e) => e.policy))];
    for (const p of policies) {
      const vals = run.entries.filter((e) => e.policy === p);
      const profit = mean(vals.map((e) => e.data.profit).filter((v): v is number => v != null));
      const overflows = mean(
        vals.map((e) => e.data.overflows).filter((v): v is number => v != null)
      );
      panels[panelId].push({
        id: `${run.path}::${p}`,
        policy: p,
        x: profit,
        y: overflows,
      });
    }
  }
  return panels;
}
