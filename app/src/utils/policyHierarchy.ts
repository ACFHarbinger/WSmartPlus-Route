/**
 * Build hierarchical policy trees for §G.2 sunburst / treemap charts.
 */

import { cityScaleLabel } from "./simMetadata";
import type { LogPathMeta, PolicyMeta } from "./simMetadata";

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
  itemStyle?: { color?: string };
  children?: HierarchyNode[];
  /** Leaf policy names under this node (for cross-filter). */
  policies: string[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function kgEfficiencyColor(kg: number, minKg: number, span: number): string {
  const t = Math.min(1, Math.max(0, (kg - minKg) / span));
  const r = Math.round(30 + t * 22);
  const g = Math.round(27 + t * 184);
  const b = Math.round(75 + t * 78);
  return `rgb(${r},${g},${b})`;
}

/** Inner = city/scale · middle = selection strategy · outer = constructor. */
export function buildPolicyHierarchy(
  policies: string[],
  stats: Record<string, PolicyAgg>,
  policyMeta: Record<string, PolicyMeta>,
  logMeta: LogPathMeta
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

  const profitSum = (ps: string[]) =>
    ps.reduce((s, p) => s + Math.max(mean(stats[p].profit), 0), 0);

  const strategyChildren: HierarchyNode[] = [];
  for (const [strat, ctorMap] of strategyMap) {
    const stratPolicies = [...ctorMap.values()].flat();
    const ctorChildren: HierarchyNode[] = [];
    for (const [ctor, ps] of ctorMap) {
      const ctorProfit = profitSum(ps);
      const ctorKg = mean(ps.map((p) => mean(stats[p]["kg/km"])));
      ctorChildren.push({
        name: ctor,
        value: Math.max(ctorProfit, 0.01),
        itemStyle: { color: kgEfficiencyColor(ctorKg, minKg, kgSpan) },
        policies: ps,
      });
    }
    strategyChildren.push({
      name: strat,
      value: Math.max(profitSum(stratPolicies), 0.01),
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

export function enrichDrillChildren(
  children: ReturnType<typeof childrenAtPath>,
  stats: Record<string, PolicyAgg>
): ReturnType<typeof childrenAtPath> {
  return children.map((c) => {
    const kg = mean(c.policies.flatMap((p) => stats[p]["kg/km"]));
    const ov = mean(c.policies.flatMap((p) => stats[p].overflows));
    return { ...c, kgkm: kg, overflows: ov };
  });
}
