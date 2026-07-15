/**
 * Bipartite attention graph on bin coordinates (§G.5.3).
 * ECharts graph overlay — edge opacity ∝ attention weight magnitude.
 */

import { attentionWeightDisplay } from "./chartLogScale";

export interface GraphCoord {
  lat: number;
  lng: number;
}

export interface AttentionGraphOpts {
  title?: string;
  theme?: "dark" | "light";
  queryRow?: number;
  topK?: number;
  sparseValues?: number[][];
  logScale?: boolean;
}

function normalizeCoords(coords: GraphCoord[]): Array<{ x: number; y: number }> {
  const lats = coords.map((c) => c.lat);
  const lngs = coords.map((c) => c.lng);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);
  const latSpan = maxLat - minLat || 1;
  const lngSpan = maxLng - minLng || 1;
  return coords.map((c) => ({
    x: ((c.lng - minLng) / lngSpan) * 100,
    y: ((c.lat - minLat) / latSpan) * 100,
  }));
}

/** Build ECharts graph series: query node → key nodes with opacity ∝ weight. */
export function buildAttentionGraphOption(
  coords: GraphCoord[],
  values: number[][],
  opts: AttentionGraphOpts = {}
): Record<string, unknown> | null {
  const { title, theme = "dark", queryRow = 0, topK = 24, logScale = false } = opts;
  const grid = opts.sparseValues ?? values;
  if (!coords.length || !grid.length) return null;

  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const qRow = Math.min(Math.max(0, queryRow), rows - 1);
  const nodeCount = Math.min(coords.length, Math.max(rows, cols));
  if (nodeCount < 2) return null;

  const positions = normalizeCoords(coords.slice(0, nodeCount));
  const row = grid[qRow] ?? [];
  const indexed = row
    .slice(0, nodeCount)
    .map((raw, i) => {
      const v = Number.isFinite(raw) ? Math.max(0, raw) : 0;
      return { raw: v, display: attentionWeightDisplay(v, logScale), i };
    })
    .sort((a, b) => b.display - a.display);
  const maxW = indexed[0]?.display || 1;
  const keep = new Set(indexed.slice(0, topK).filter((x) => x.display > 0).map((x) => x.i));

  const nodes = positions.map((p, i) => ({
    id: String(i),
    name: i === 0 ? "Depot" : `Bin ${i}`,
    x: p.x,
    y: p.y,
    symbolSize: i === qRow ? 14 : i === 0 ? 12 : 8,
    itemStyle: {
      color:
        i === qRow
          ? "#f59e0b"
          : i === 0
            ? "#6366f1"
            : theme === "dark"
              ? "#38bdf8"
              : "#0284c7",
      borderColor: i === qRow ? "#fbbf24" : "transparent",
      borderWidth: i === qRow ? 2 : 0,
    },
    label: { show: i === 0 || i === qRow, fontSize: 8, color: "#9ca3af" },
  }));

  const links: Array<{
    source: string;
    target: string;
    value: number;
    lineStyle: { opacity: number; width: number; color: string };
  }> = [];

  for (const { raw, display, i } of indexed) {
    if (!keep.has(i) || i === qRow) continue;
    const opacity = 0.15 + 0.75 * (display / maxW);
    links.push({
      source: String(qRow),
      target: String(i),
      value: raw,
      lineStyle: {
        opacity,
        width: 1 + 3 * (display / maxW),
        color: `rgba(251, 191, 36, ${opacity})`,
      },
    });
  }

  return {
    backgroundColor: "transparent",
    title: title
      ? {
          text: logScale ? `${title} · log edges` : title,
          left: "center",
          top: 4,
          textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" },
        }
      : undefined,
    tooltip: {
      formatter: (p: { dataType?: string; data?: { name?: string; value?: number } }) => {
        if (p.dataType === "edge") {
          return `Attention ${Number(p.data?.value ?? 0).toFixed(4)}`;
        }
        return p.data?.name ?? "";
      },
    },
    series: [
      {
        type: "graph",
        layout: "none",
        coordinateSystem: null,
        roam: true,
        data: nodes,
        links,
        emphasis: { focus: "adjacency", lineStyle: { width: 4 } },
        lineStyle: { curveness: 0.12 },
      },
    ],
  };
}
