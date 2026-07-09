export interface ParetoPoint {
  id: string;
  x: number;
  y: number;
}

/** Non-dominated points when maximising x and minimising y. */
export function paretoFront(points: ParetoPoint[]): ParetoPoint[] {
  return points.filter(
    (p) =>
      !points.some(
        (q) =>
          q.id !== p.id &&
          q.x >= p.x &&
          q.y <= p.y &&
          (q.x > p.x || q.y < p.y)
      )
  );
}

/** Step-line coords for a Pareto front sorted by ascending x. */
export function paretoStepLine(front: ParetoPoint[]): Array<[number, number]> {
  if (front.length === 0) return [];
  const sorted = [...front].sort((a, b) => a.x - b.x);
  const steps: Array<[number, number]> = [];
  for (let i = 0; i < sorted.length; i++) {
    steps.push([sorted[i].x, sorted[i].y]);
    if (i < sorted.length - 1) {
      steps.push([sorted[i + 1].x, sorted[i].y]);
    }
  }
  return steps;
}
