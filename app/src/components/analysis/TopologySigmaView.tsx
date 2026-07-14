/**
 * Sigma.js WebGL topology graph — distance-matrix k-NN overlay (§G.4).
 */
import Graph from "graphology";
import Sigma from "sigma";
import { useEffect, useRef } from "react";
import type { GraphEdge, TopologyNodeMeta } from "../../utils/graphTopology";
import {
  pheromoneEdgeKey,
  topologyNodeStyle,
} from "../../utils/graphTopology";

export interface TopologySigmaOpts {
  nodeMeta: TopologyNodeMeta[];
  edges: GraphEdge[];
  positions: [number, number][];
  fillRange?: [number, number] | null;
  pheromoneWeights?: Map<string, number>;
  showPheromone?: boolean;
  theme?: "dark" | "light";
}

function normalizePositions(
  positions: [number, number][]
): Array<{ x: number; y: number }> {
  const xs = positions.map((p) => p[0]);
  const ys = positions.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  return positions.map(([x, y]) => ({
    x: ((x - minX) / spanX) * 100,
    y: ((y - minY) / spanY) * 100,
  }));
}

function buildGraph(opts: TopologySigmaOpts): Graph | null {
  const {
    nodeMeta,
    edges,
    positions,
    fillRange = null,
    pheromoneWeights,
    showPheromone = false,
    theme = "dark",
  } = opts;
  if (!nodeMeta.length) return null;

  const norm = normalizePositions(positions);
  const graph = new Graph({ multi: false, type: "undirected" });

  for (const meta of nodeMeta) {
    const pos = norm[meta.matrixIndex] ?? { x: 0, y: 0 };
    const style = topologyNodeStyle(meta, fillRange);
    graph.addNode(String(meta.matrixIndex), {
      label: meta.nodeId === 0 ? "Depot" : `#${meta.nodeId}`,
      x: pos.x,
      y: -pos.y,
      size: style.size * 0.45,
      color: style.color,
    });
  }

  const dists = edges.map((e) => e.distance);
  const minD = dists.length ? Math.min(...dists) : 0;
  const maxD = dists.length ? Math.max(...dists) : 1;
  const phMax = showPheromone && pheromoneWeights?.size
    ? Math.max(...pheromoneWeights.values(), 1e-9)
    : 1;

  for (const edge of edges) {
    const src = String(edge.source);
    const tgt = String(edge.target);
    if (!graph.hasNode(src) || !graph.hasNode(tgt)) continue;
    const edgeId = pheromoneEdgeKey(edge.source, edge.target);
    if (graph.hasEdge(edgeId)) continue;

    const tNorm = maxD > minD ? (maxD - edge.distance) / (maxD - minD) : 0.5;
    const ph = showPheromone ? (pheromoneWeights?.get(edgeId) ?? 0) / phMax : 0;
    const baseOpacity = theme === "dark" ? 0.22 : 0.35;
    const opacity = ph > 0 ? 0.15 + ph * 0.75 : baseOpacity;
    const width = ph > 0 ? 0.6 + ph * 3 : 0.4 + tNorm * 1.6;
    const color =
      ph > 0
        ? `rgba(251, 191, 36, ${opacity})`
        : theme === "dark"
          ? `rgba(100, 116, 139, ${opacity})`
          : `rgba(71, 85, 105, ${opacity})`;

    graph.addEdgeWithKey(edgeId, src, tgt, { size: width, color, weight: ph || tNorm });
  }

  return graph;
}

export function TopologySigmaView({
  nodeMeta,
  edges,
  positions,
  fillRange = null,
  pheromoneWeights,
  showPheromone = false,
  theme = "dark",
}: TopologySigmaOpts) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const graph = buildGraph({
      nodeMeta,
      edges,
      positions,
      fillRange,
      pheromoneWeights,
      showPheromone,
      theme,
    });
    if (!graph) return;

    sigmaRef.current?.kill();
    const sigma = new Sigma(graph, container, {
      renderEdgeLabels: false,
      labelDensity: 0.05,
      labelGridCellSize: 80,
      defaultNodeType: "circle",
      zIndex: true,
    });
    sigmaRef.current = sigma;

    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
  }, [nodeMeta, edges, positions, fillRange, pheromoneWeights, showPheromone, theme]);

  return (
    <div
      ref={containerRef}
      className="w-full h-[360px] rounded-lg border border-canvas-border bg-canvas-elevated"
    />
  );
}
