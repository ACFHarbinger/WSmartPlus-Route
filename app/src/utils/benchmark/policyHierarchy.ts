/**
 * Build hierarchical policy trees for §G.2 sunburst / treemap charts.
 */

import {
  cityScaleLabel,
  parseLogPath,
  SELECTION_STRATEGY_LEGEND,
  selectionStrategyColor,
} from "../sim/simMetadata";
import type { LogPathMeta, PolicyMeta } from "../sim/simMetadata";

export { cityScaleLabel };

export interface PolicyAgg {
  profit: number[];
  km: number[];
  overflows: number[];
  kg: number[];
  "kg/km": number[];
}

export interface HierarchyNode {
  name: string;
  value: number;
  itemStyle?: { color?: string; borderColor?: string; borderWidth?: number };
  children?: HierarchyNode[];
  /** Leaf policy names under this node (for cross-filter). */
  policies: string[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export type HierarchyColorMode = "kgkm" | "overflows";

function kgEfficiencyColor(kg: number, minKg: number, span: number): string {
  const t = Math.min(1, Math.max(0, (kg - minKg) / span));
  const r = Math.round(30 + t * 22);
  const g = Math.round(27 + t * 184);
  const b = Math.round(75 + t * 78);
  return `rgb(${r},${g},${b})`;
}

/** Green (0 overflows) → amber → red (high overflows). */
function overflowsColor(ov: number, minOv: number, span: number): string {
  const t = Math.min(1, Math.max(0, (ov - minOv) / span));
  const r = Math.round(34 + t * 205);
  const g = Math.round(197 - t * 132);
  const b = Math.round(94 - t * 60);
  return `rgb(${r},${g},${b})`;
}

const STRATEGY_LABELS = new Set<string>(SELECTION_STRATEGY_LEGEND);

/** Drill-down bar fill: strategy chips at strategy depth, kg/km or overflow gradient at constructor depth. */
export function resolveDrillBarColor(
  name: string,
  policies: string[],
  stats: Record<string, PolicyAgg>,
  colorMode: HierarchyColorMode
): string {
  if (STRATEGY_LABELS.has(name)) {
    return selectionStrategyColor(name);
  }

  if (!policies.length) return "#6366f1";

  const kgkmValues = policies.map((p) => mean(stats[p]["kg/km"]));
  const minKg = Math.min(...kgkmValues, 0);
  const maxKg = Math.max(...kgkmValues, 0.01);
  const kgSpan = maxKg - minKg || 1;

  const overflowValues = policies.map((p) => mean(stats[p].overflows));
  const minOv = Math.min(...overflowValues, 0);
  const maxOv = Math.max(...overflowValues, 0.01);
  const ovSpan = maxOv - minOv || 1;

  if (colorMode === "overflows") {
    const ov = mean(policies.map((p) => mean(stats[p].overflows)));
    return overflowsColor(ov, minOv, ovSpan);
  }

  const kg = mean(policies.map((p) => mean(stats[p]["kg/km"])));
  return kgEfficiencyColor(kg, minKg, kgSpan);
}

/** Inner = city/scale · middle = selection strategy · outer = constructor. */
export function buildPolicyHierarchy(
  policies: string[],
  stats: Record<string, PolicyAgg>,
  policyMeta: Record<string, PolicyMeta>,
  logMeta: LogPathMeta,
  colorMode: HierarchyColorMode = "kgkm"
): HierarchyNode[] {
  const rootLabel = cityScaleLabel(logMeta);

  const strategyMap = new Map<string, Map<string, string[]>>();
  for (const p of policies) {
    const strat = policyMeta[p]?.selectionStrategy ?? "Other";
    const ctor = policyMeta[p]?.constructor ?? "Other";
    if (!strategyMap.has(strat)) strategyMap.set(strat, new Map());
    const ctorMap = strategyMap.get(strat)!;
    if (!ctorMap.has(ctor)) ctorMap.set(ctor, []);
    ctorMap.get(ctor)!.push(p);
  }

  const kgkmValues = policies.map((p) => mean(stats[p]["kg/km"]));
  const minKg = Math.min(...kgkmValues, 0);
  const maxKg = Math.max(...kgkmValues, 0.01);
  const kgSpan = maxKg - minKg || 1;

  const overflowValues = policies.map((p) => mean(stats[p].overflows));
  const minOv = Math.min(...overflowValues, 0);
  const maxOv = Math.max(...overflowValues, 0.01);
  const ovSpan = maxOv - minOv || 1;

  const segmentColor = (ps: string[]) => {
    if (colorMode === "overflows") {
      const ov = mean(ps.map((p) => mean(stats[p].overflows)));
      return overflowsColor(ov, minOv, ovSpan);
    }
    const kg = mean(ps.map((p) => mean(stats[p]["kg/km"])));
    return kgEfficiencyColor(kg, minKg, kgSpan);
  };

  const profitSum = (ps: string[]) =>
    ps.reduce((s, p) => s + Math.max(mean(stats[p].profit), 0), 0);

  const strategyChildren: HierarchyNode[] = [];
  for (const [strat, ctorMap] of strategyMap) {
    const stratPolicies = [...ctorMap.values()].flat();
    const ctorChildren: HierarchyNode[] = [];
    for (const [ctor, ps] of ctorMap) {
      const ctorProfit = profitSum(ps);
      ctorChildren.push({
        name: ctor,
        value: Math.max(ctorProfit, 0.01),
        itemStyle: { color: segmentColor(ps) },
        policies: ps,
      });
    }
    strategyChildren.push({
      name: strat,
      value: Math.max(profitSum(stratPolicies), 0.01),
      itemStyle: {
        color: segmentColor(stratPolicies),
        borderColor: selectionStrategyColor(strat),
        borderWidth: 2,
      },
      children: ctorChildren,
      policies: stratPolicies,
    });
  }

  return [
    {
      name: rootLabel,
      value: Math.max(profitSum(policies), 0.01),
      children: strategyChildren,
      policies,
    },
  ];
}

export interface PortfolioHierarchyRun {
  path: string;
  policies: string[];
  stats: Record<string, PolicyAgg>;
  policyMeta: Record<string, PolicyMeta>;
}

/** Multi-root sunburst: one city/scale ring per loaded simulation log (§G.2 portfolio). */
export function buildPortfolioHierarchy(
  runs: PortfolioHierarchyRun[],
  colorMode: HierarchyColorMode = "kgkm"
): HierarchyNode[] {
  return runs.flatMap((run) =>
    buildPolicyHierarchy(
      run.policies,
      run.stats,
      run.policyMeta,
      parseLogPath(run.path),
      colorMode
    )
  );
}

/** Find node policies along a breadcrumb path (e.g. RM-100 → LA → SWC-TCF). */
export function policiesAtPath(nodes: HierarchyNode[], path: string[]): string[] {
  if (path.length === 0) return [];
  let current: HierarchyNode | undefined = nodes.find((n) => n.name === path[0]);
  for (let i = 1; i < path.length && current; i++) {
    current = current.children?.find((c) => c.name === path[i]);
  }
  return current?.policies ?? [];
}

/** Direct children at a drill path for bar-chart drill-down. */
export function childrenAtPath(
  nodes: HierarchyNode[],
  path: string[]
): Array<{ name: string; policies: string[]; profit: number; kgkm: number; overflows: number }> {
  if (path.length === 0) {
    const root = nodes[0];
    return (root?.children ?? []).map((c) => summarizeNode(c, nodes));
  }
  let current: HierarchyNode | undefined = nodes.find((n) => n.name === path[0]);
  for (let i = 1; i < path.length && current; i++) {
    current = current.children?.find((c) => c.name === path[i]);
  }
  return (current?.children ?? [current].filter(Boolean) as HierarchyNode[]).map((c) =>
    summarizeNode(c, nodes)
  );
}

function summarizeNode(
  node: HierarchyNode,
  _roots: HierarchyNode[]
): { name: string; policies: string[]; profit: number; kgkm: number; overflows: number } {
  return {
    name: node.name,
    policies: node.policies,
    profit: node.value,
    kgkm: 0,
    overflows: 0,
  };
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

export interface EnrichedDrillChild {
  name: string;
  policies: string[];
  profit: number;
  kgkm: number;
  overflows: number;
  profitStd: number;
  distSpread: number;
}

export function enrichDrillChildren(
  children: ReturnType<typeof childrenAtPath>,
  stats: Record<string, PolicyAgg>,
  policyMeta?: Record<string, { distribution: string }>
): EnrichedDrillChild[] {
  return children.map((c) => {
    const profits = c.policies.flatMap((p) => stats[p].profit);
    const kg = mean(c.policies.flatMap((p) => stats[p]["kg/km"]));
    const ov = mean(c.policies.flatMap((p) => stats[p].overflows));

    let distSpread = 0;
    if (policyMeta) {
      const emp = c.policies.filter((p) => /emp/i.test(policyMeta[p]?.distribution ?? ""));
      const gamma = c.policies.filter((p) => /gamma/i.test(policyMeta[p]?.distribution ?? ""));
      if (emp.length && gamma.length) {
        const empMean = mean(emp.flatMap((p) => stats[p].profit));
        const gammaMean = mean(gamma.flatMap((p) => stats[p].profit));
        distSpread = Math.abs(empMean - gammaMean);
      }
    }

    return {
      ...c,
      kgkm: kg,
      overflows: ov,
      profitStd: std(profits),
      distSpread,
    };
  });
}
