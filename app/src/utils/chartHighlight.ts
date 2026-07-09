/** Cross-filter highlight helpers for §G.1 dashboard brushing. */

export function isHighlighted(policy: string, brushed: string[] | null): boolean {
  return !brushed || brushed.length === 0 || brushed.includes(policy);
}

export function barOpacity(policy: string, brushed: string[] | null): number {
  return isHighlighted(policy, brushed) ? 1 : 0.2;
}

export function toggleBrush(current: string[] | null, policy: string): string[] {
  const active = current ?? [];
  if (active.includes(policy)) {
    const next = active.filter((p) => p !== policy);
    return next.length === 0 ? [] : next;
  }
  return [...active, policy];
}
