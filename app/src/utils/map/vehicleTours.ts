/**
 * Split simulation day tours into per-vehicle segments (§G.3.2).
 */

import type { SimDayData } from "../../types";

/** ColorBrewer Set1 — mirrors logic/src/constants/dashboard.py ROUTE_COLORS */
export const VEHICLE_COLORS_RGB: [number, number, number][] = [
  [228, 26, 28],
  [55, 126, 184],
  [77, 175, 74],
  [152, 78, 163],
  [255, 127, 0],
  [255, 255, 51],
  [166, 86, 40],
  [247, 129, 191],
];

function isDepot(node: number): boolean {
  return node <= 0;
}

/** Split raw tour node list (1-based customers, 0/-1 depot) into 0-based bin index segments. */
export function splitVehicleTourIndices(data: SimDayData): number[][] {
  const { tour, tour_indices } = data;

  if (tour?.length) {
    const nodes = tour.map((stop) => (typeof stop === "number" ? stop : stop.id));
    const depotSplits = nodes.filter(isDepot).length;
    if (depotSplits >= 2) {
      const segments: number[][] = [];
      let current: number[] = [];
      for (const node of nodes) {
        if (isDepot(node)) {
          if (current.length) {
            segments.push(current.map((n) => (n > 0 ? n - 1 : n)).filter((n) => n >= 0));
            current = [];
          }
        } else {
          current.push(node);
        }
      }
      if (current.length) {
        segments.push(current.map((n) => (n > 0 ? n - 1 : n)).filter((n) => n >= 0));
      }
      if (segments.length > 1) return segments;
    }
  }

  if (tour_indices?.length) return [tour_indices];
  return [];
}

export function vehicleCount(data: SimDayData): number {
  return splitVehicleTourIndices(data).length;
}
