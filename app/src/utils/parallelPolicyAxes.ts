/**
 * Ten-axis parallel-coordinates schema for policy exploration (§G.1.4).
 */
import type { LogPathMeta, PolicyMeta } from "./simMetadata";

export interface PolicyStatsSlice {
  profit: number[];
  km: number[];
  overflows: number[];
  "kg/km": number[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.filter((v) => v && v !== "—"))].sort();
}

export interface PolicyParallelAxis {
  dim: number;
  name: string;
  type: "category" | "value";
  data?: string[];
  max?: number;
}

export interface PolicyParallelRow {
  name: string;
  value: Array<string | number>;
}

export interface PolicyParallelBuild {
  axes: PolicyParallelAxis[];
  rows: PolicyParallelRow[];
  overflowDim: number;
}

/** Build city · N · dist · improver · strategy · constructor · overflows · kg/km · km · profit axes. */
export function buildPolicyParallelAxes(
  policies: string[],
  stats: Record<string, PolicyStatsSlice>,
  policyMeta: Record<string, PolicyMeta>,
  logMeta: LogPathMeta
): PolicyParallelBuild {
  const cityLabel =
    logMeta.cityShort && logMeta.scale
      ? `${logMeta.cityShort}-${logMeta.scale}`
      : logMeta.city ?? "—";

  const nValue = logMeta.scale ?? 100;
  const distCategories = uniqueSorted(
    policies.map((p) => policyMeta[p]?.distribution ?? "—")
  );
  const improverCategories = uniqueSorted(
    policies.map((p) => policyMeta[p]?.improver ?? "—")
  );
  const strategyCategories = uniqueSorted(
    policies.map((p) => policyMeta[p]?.selectionStrategy ?? "—")
  );
  const constructorCategories = uniqueSorted(
    policies.map((p) => policyMeta[p]?.constructor ?? "—")
  );

  const overflowValues = policies.map((p) => mean(stats[p].overflows));
  const kgkmValues = policies.map((p) => mean(stats[p]["kg/km"]));
  const kmValues = policies.map((p) => mean(stats[p].km));
  const profitValues = policies.map((p) => mean(stats[p].profit));

  const axes: PolicyParallelAxis[] = [
    { dim: 0, name: "City", type: "category", data: [cityLabel] },
    { dim: 1, name: "N", type: "value", max: Math.max(nValue, 1) * 1.1 },
    { dim: 2, name: "Dist", type: "category", data: distCategories },
    { dim: 3, name: "Improver", type: "category", data: improverCategories },
    { dim: 4, name: "Strategy", type: "category", data: strategyCategories },
    { dim: 5, name: "Constructor", type: "category", data: constructorCategories },
    {
      dim: 6,
      name: "Overflows",
      type: "value",
      max: Math.max(...overflowValues, 1) * 1.1,
    },
    {
      dim: 7,
      name: "kg/km",
      type: "value",
      max: Math.max(...kgkmValues, 0.01) * 1.1,
    },
    {
      dim: 8,
      name: "km",
      type: "value",
      max: Math.max(...kmValues, 1) * 1.1,
    },
    {
      dim: 9,
      name: "Profit",
      type: "value",
      max: Math.max(...profitValues, 1) * 1.1,
    },
  ];

  const rows: PolicyParallelRow[] = policies.map((p) => ({
    name: p,
    value: [
      cityLabel,
      nValue,
      policyMeta[p]?.distribution ?? "—",
      policyMeta[p]?.improver ?? "—",
      policyMeta[p]?.selectionStrategy ?? "—",
      policyMeta[p]?.constructor ?? "—",
      mean(stats[p].overflows),
      mean(stats[p]["kg/km"]),
      mean(stats[p].km),
      mean(stats[p].profit),
    ],
  }));

  return { axes, rows, overflowDim: 6 };
}
