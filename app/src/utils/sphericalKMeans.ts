/**
 * Spherical k-means for attention query-row clustering (§G.5.3).
 */

export const CLUSTER_PALETTE = [
  "#3b82f6",
  "#22c55e",
  "#f59e0b",
  "#ec4899",
  "#8b5cf6",
  "#06b6d4",
  "#ef4444",
  "#a3e635",
];

function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function normalizeRow(row: number[]): number[] {
  const norm = Math.sqrt(row.reduce((s, x) => s + x * x, 0)) || 1;
  return row.map((x) => x / norm);
}

/** Assign each row to one of *k* clusters on the unit sphere. */
export function sphericalKMeans(rows: number[][], k: number, maxIter = 24): number[] {
  const n = rows.length;
  if (n === 0) return [];
  const clusters = Math.max(1, Math.min(k, n));
  if (clusters === 1) return Array(n).fill(0);

  const normalized = rows.map(normalizeRow);
  const centroids: number[][] = [];
  centroids.push([...normalized[0]]);

  while (centroids.length < clusters) {
    const dists = normalized.map((v) => {
      const sims = centroids.map((c) => dot(v, c));
      return 1 - Math.max(...sims);
    });
    const total = dists.reduce((a, b) => a + b, 0) || 1;
    let r = Math.random() * total;
    let pick = 0;
    for (let i = 0; i < n; i++) {
      r -= dists[i];
      if (r <= 0) {
        pick = i;
        break;
      }
    }
    centroids.push([...normalized[pick]]);
  }

  let labels = Array(n).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    labels = normalized.map((v) => {
      let best = 0;
      let bestSim = -Infinity;
      centroids.forEach((c, j) => {
        const sim = dot(v, c);
        if (sim > bestSim) {
          bestSim = sim;
          best = j;
        }
      });
      return best;
    });

    const next: number[][] = [];
    for (let j = 0; j < clusters; j++) {
      const members = normalized.filter((_, i) => labels[i] === j);
      if (!members.length) {
        next.push([...centroids[j]]);
        continue;
      }
      const dim = members[0].length;
      const sum = Array(dim).fill(0);
      for (const m of members) {
        for (let d = 0; d < dim; d++) sum[d] += m[d];
      }
      const norm = Math.sqrt(sum.reduce((s, x) => s + x * x, 0)) || 1;
      next.push(sum.map((x) => x / norm));
    }
    centroids.splice(0, centroids.length, ...next);
  }

  return labels;
}

/** Reorder matrix rows by cluster id for banded heatmap display. */
export function reorderRowsByClusters(
  values: number[][],
  labels: number[]
): { values: number[][]; bandSplits: number[]; labels: number[] } {
  const indexed = labels.map((l, i) => ({ l, i }));
  indexed.sort((a, b) => a.l - b.l || a.i - b.i);
  const reordered = indexed.map(({ i }) => values[i]);
  const reorderedLabels = indexed.map(({ l }) => l);
  const bandSplits: number[] = [];
  for (let p = 1; p < indexed.length; p++) {
    if (indexed[p].l !== indexed[p - 1].l) bandSplits.push(p);
  }
  return { values: reordered, bandSplits, labels: reorderedLabels };
}
