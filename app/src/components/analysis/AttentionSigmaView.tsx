/**
 * Sigma.js WebGL bipartite attention graph on bin coordinates (§G.5.3).
 */
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import { useEffect, useRef } from "react";
import type { GraphCoord } from "../../utils/attentionGraph";

export interface AttentionSigmaOpts {
  coords: GraphCoord[];
  values: number[][];
  queryRow?: number;
  topK?: number;
  sparseValues?: number[][];
  theme?: "dark" | "light";
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

function buildGraph(opts: AttentionSigmaOpts): Graph | null {
  const { coords, values, queryRow = 0, topK = 24, theme = "dark" } = opts;
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
    .map((v, i) => ({ v: Number.isFinite(v) ? Math.max(0, v) : 0, i }))
    .sort((a, b) => b.v - a.v);
  const maxW = indexed[0]?.v || 1;
  const keep = new Set(indexed.slice(0, topK).filter((x) => x.v > 0).map((x) => x.i));

  const graph = new Graph({ multi: false, type: "directed" });
  const idleColor = theme === "dark" ? "#38bdf8" : "#0284c7";

  for (let i = 0; i < nodeCount; i += 1) {
    graph.addNode(String(i), {
      label: i === 0 ? "Depot" : i === qRow ? `Q${i}` : `Bin ${i}`,
      x: positions[i].x,
      y: -positions[i].y,
      size: i === qRow ? 10 : i === 0 ? 8 : 5,
      color: i === qRow ? "#f59e0b" : i === 0 ? "#6366f1" : idleColor,
    });
  }

  for (const { v, i } of indexed) {
    if (!keep.has(i) || i === qRow) continue;
    const opacity = 0.2 + 0.75 * (v / maxW);
    const edgeId = `e-${qRow}-${i}`;
    if (!graph.hasEdge(edgeId)) {
      graph.addEdgeWithKey(edgeId, String(qRow), String(i), {
        size: 0.5 + 2.5 * (v / maxW),
        color: `rgba(251, 191, 36, ${opacity})`,
        weight: v,
      });
    }
  }

  forceAtlas2.assign(graph, { iterations: 40, settings: { gravity: 0.4, scalingRatio: 4 } });
  return graph;
}

export function AttentionSigmaView({
  coords,
  values,
  queryRow = 0,
  topK = 24,
  sparseValues,
  theme = "dark",
}: AttentionSigmaOpts) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const graph = buildGraph({ coords, values, queryRow, topK, sparseValues, theme });
    if (!graph) return;

    sigmaRef.current?.kill();
    const sigma = new Sigma(graph, container, {
      renderEdgeLabels: false,
      labelDensity: 0.07,
      labelGridCellSize: 60,
      defaultEdgeType: "arrow",
      defaultNodeType: "circle",
      zIndex: true,
    });
    sigmaRef.current = sigma;

    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
  }, [coords, values, sparseValues, queryRow, topK, theme]);

  return (
    <div
      ref={containerRef}
      className="w-full h-[400px] rounded-lg border border-canvas-border bg-canvas-elevated"
    />
  );
}
