/**
 * ECharts heatmap builders for tensor slices and loss landscapes (§G.5).
 */

export type AttentionRole = "query" | "key" | "value" | "weights";

const ROLE_PALETTES: Record<AttentionRole, string[]> = {
  query: ["#0c4a6e", "#0369a1", "#0ea5e9", "#7dd3fc"],
  key: ["#14532d", "#15803d", "#22c55e", "#86efac"],
  value: ["#78350f", "#b45309", "#f59e0b", "#fde68a"],
  weights: ["#1e3a8a", "#6366f1", "#fbbf24", "#ef4444"],
};

/** Classify tensor key as Q/K/V projection or fused attention weights. */
export function classifyAttentionRole(key: string): AttentionRole {
  const k = key.toLowerCase();
  if (/\bquery\b|_q_|\.q\.|attn_q|q_proj|wq\b/.test(k)) return "query";
  if (/\bkey\b|_k_|\.k\.|attn_k|k_proj|wk\b/.test(k) && !/keyboard|keypad/.test(k)) return "key";
  if (/\bvalue\b|_v_|\.v\.|attn_v|v_proj|wv\b/.test(k)) return "value";
  return "weights";
}

export function rolePalette(role: AttentionRole): string[] {
  return ROLE_PALETTES[role];
}

export function groupAttentionKeys(arrays: { key: string; shape: number[] }[]): {
  query: string[];
  key: string[];
  value: string[];
  weights: string[];
} {
  const groups = { query: [] as string[], key: [] as string[], value: [] as string[], weights: [] as string[] };
  for (const a of arrays) {
    const role = classifyAttentionRole(a.key);
    groups[role].push(a.key);
  }
  return groups;
}

export function buildMatrixHeatmapOption(
  values: number[][],
  opts: {
    title?: string;
    min?: number;
    max?: number;
    theme?: "dark" | "light";
    xLabel?: string;
    yLabel?: string;
    attentionRole?: AttentionRole;
  } = {}
): Record<string, unknown> {
  const {
    title,
    min,
    max,
    theme = "dark",
    xLabel = "Col",
    yLabel = "Row",
    attentionRole,
  } = opts;
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const flat = values.flat();
  const dataMin = min ?? Math.min(...flat);
  const dataMax = max ?? Math.max(...flat);

  const data: Array<[number, number, number | string]> = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      data.push([c, r, Number.isFinite(v) ? v : "-"]);
    }
  }

  return {
    backgroundColor: "transparent",
    title: title
      ? {
          text: title,
          left: "center",
          top: 4,
          textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" },
        }
      : undefined,
    tooltip: {
      position: "top",
      formatter: (p: { value?: [number, number, number] }) => {
        const v = p.value;
        if (!v) return "";
        return `${yLabel} ${v[1]} · ${xLabel} ${v[0]}<br/>${Number(v[2]).toFixed(4)}`;
      },
    },
    grid: { left: 48, right: 56, top: title ? 36 : 16, bottom: 36 },
    xAxis: {
      type: "category",
      name: xLabel,
      data: Array.from({ length: cols }, (_, i) => String(i)),
      splitArea: { show: true },
      axisLabel: { color: "#9090b0", fontSize: 9, interval: Math.max(0, Math.floor(cols / 12)) },
    },
    yAxis: {
      type: "category",
      name: yLabel,
      data: Array.from({ length: rows }, (_, i) => String(i)),
      splitArea: { show: true },
      axisLabel: { color: "#9090b0", fontSize: 9, interval: Math.max(0, Math.floor(rows / 12)) },
    },
    visualMap: {
      min: dataMin,
      max: dataMax,
      calculable: true,
      orient: "vertical",
      right: 4,
      top: "center",
      inRange: {
        color: attentionRole ? rolePalette(attentionRole) : ROLE_PALETTES.weights,
      },
      textStyle: { color: "#9090b0", fontSize: 9 },
    },
    series: [
      {
        type: "heatmap",
        data,
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: "rgba(0,0,0,0.4)" } },
      },
    ],
  };
}

/** Pick arrays that look like attention weight matrices. */
export function suggestAttentionKeys(arrays: { key: string; shape: number[] }[]): string[] {
  return arrays
    .filter((a) => {
      const rank = a.shape.length;
      if (rank < 2) return false;
      const last = a.shape[rank - 1];
      const prev = a.shape[rank - 2];
      return last >= 4 && prev >= 4 && /attn|attention|weights|alpha/i.test(a.key);
    })
    .map((a) => a.key);
}

export function leadingIndexCount(shape: number[]): number {
  return Math.max(0, shape.length - 2);
}

export function defaultIndices(shape: number[]): number[] {
  return Array.from({ length: leadingIndexCount(shape) }, () => 0);
}

/** Guess which leading dimension is the attention head axis. */
export function detectHeadAxis(shape: number[], key: string): number | null {
  const leading = leadingIndexCount(shape);
  if (leading < 1) return null;
  if (/head/i.test(key)) {
    const idx = shape.findIndex((_, i) => i < leading);
    return idx >= 0 ? idx : 0;
  }
  // Common layouts: (H, N, N) or (L, H, N, N) — first small leading dim is often heads.
  for (let i = 0; i < leading; i++) {
    const dim = shape[i];
    if (dim >= 2 && dim <= 32 && /attn|attention|weights|alpha/i.test(key)) {
      return i;
    }
  }
  return null;
}

/** Keep only top-k values per query row (sparse attention, §G.5.3). */
export function applySparseTopK(values: number[][], k: number): number[][] {
  if (k <= 0) return values;
  return values.map((row) => {
    const indexed = row
      .map((v, i) => ({ v: Number.isFinite(v) ? v : -Infinity, i }))
      .sort((a, b) => b.v - a.v);
    const keep = new Set(indexed.slice(0, k).map((x) => x.i));
    return row.map((v, i) => (keep.has(i) ? v : 0));
  });
}

/** Difference heatmap for overlay-compare mode (current − baseline). */
export function diffMatrices(a: number[][], b: number[][]): number[][] {
  const rows = Math.min(a.length, b.length);
  const cols = Math.min(a[0]?.length ?? 0, b[0]?.length ?? 0);
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push((a[r]?.[c] ?? 0) - (b[r]?.[c] ?? 0));
    }
    out.push(row);
  }
  return out;
}
