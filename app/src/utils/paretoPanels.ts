/**
 * Multi-run Pareto panel classification (§G.1.2 four-panel layout).
 */

import type { LogPathMeta } from "./simMetadata";

export interface ParetoPanelDef {
  id: string;
  label: string;
  match: (meta: LogPathMeta) => boolean;
}

export const PARETO_PANELS: ParetoPanelDef[] = [
  {
    id: "gamma3-ftsp",
    label: "Gamma-3 / FTSP",
    match: (m) => /gamma/i.test(m.distributionKey ?? "") && m.improver === "FTSP",
  },
  {
    id: "emp-ftsp",
    label: "Empirical / FTSP",
    match: (m) => /emp/i.test(m.distributionKey ?? "") && m.improver === "FTSP",
  },
  {
    id: "gamma3-cls",
    label: "Gamma-3 / CLS",
    match: (m) => /gamma/i.test(m.distributionKey ?? "") && m.improver === "CLS",
  },
  {
    id: "emp-cls",
    label: "Empirical / CLS",
    match: (m) => /emp/i.test(m.distributionKey ?? "") && m.improver === "CLS",
  },
];

export function panelForRun(meta: LogPathMeta): string | null {
  const hit = PARETO_PANELS.find((p) => p.match(meta));
  return hit?.id ?? null;
}
