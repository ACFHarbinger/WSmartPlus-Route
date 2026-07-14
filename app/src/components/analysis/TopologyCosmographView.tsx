/**
 * Cosmograph-style dense-graph WebGL renderer — Sigma.js point mode (§G.4).
 */
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import { useEffect, useRef } from "react";
import type { GraphEdge, TopologyNodeMeta } from "../../utils/graphTopology";
import {
  pheromoneEdgeKey,
  topologyNodeStyle,
} from "../../utils/graphTopology";

export interface TopologyCosmographOpts {
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
    x: ((x - minX) / spanX) * 120,
    y: ((y - minY) / spanY) * 120,
  }));
}

function buildDenseGraph(opts: TopologyCosmographOpts): Graph | null {
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
      x: pos.x,
      y: -pos.y,
      size: Math.max(1.2, style.size * 0.22),
      color: style.dimmed ? `${style.color}55` : style.color,
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
    const baseOpacity = theme === "dark" ? 0.08 : 0.14;
    const opacity = ph > 0 ? 0.12 + ph * 0.55 : baseOpacity;
    const width = ph > 0 ? 0.3 + ph * 1.4 : 0.15 + tNorm * 0.6;
    const color =
      ph > 0
        ? `rgba(251, 191, 36, ${opacity})`
        : theme === "dark"
          ? `rgba(71, 85, 105, ${opacity})`
          : `rgba(148, 163, 184, ${opacity})`;

    graph.addEdgeWithKey(edgeId, src, tgt, { size: width, color, weight: ph || tNorm });
  }

  forceAtlas2.assign(graph, {
    iterations: 80,
    settings: {
      gravity: 1.2,
      scalingRatio: 12,
      strongGravityMode: true,
      barnesHutOptimize: true,
      barnesHutTheta: 0.5,
    },
  });

  return graph;
}

export function TopologyCosmographView({
  nodeMeta,
  edges,
  positions,
  fillRange = null,
  pheromoneWeights,
  showPheromone = false,
  theme = "dark",
}: TopologyCosmographOpts) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const graph = buildDenseGraph({
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
      renderLabels: false,
      labelDensity: 0,
      hideEdgesOnMove: true,
      hideLabelsOnMove: true,
      defaultNodeType: "point",
      defaultEdgeType: "line",
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
