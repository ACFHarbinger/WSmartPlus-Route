const START = performance.now();
const marks: Record<string, number> = { start: START };

/** Record a startup milestone (§G.7 performance probe). */
export function markStartup(label: string): void {
  marks[label] = performance.now();
}

export function getStartupElapsed(label: string): number | null {
  const t = marks[label];
  return t != null ? Math.round(t - START) : null;
}

export function getStartupMarks(): Readonly<Record<string, number>> {
  return marks;
}
