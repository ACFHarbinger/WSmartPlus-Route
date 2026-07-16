/**
 * Resolve bin positions for map layers — geographic or abstract circular layout.
 */

import type { BinCoord } from "../../types";

export function resolveBinPositions(bins: BinCoord[]): {
  posById: Map<number, [number, number]>;
  hasGeo: boolean;
} {
  const hasGeo = bins.some((b) => b.lat != null && b.lng != null);
  const posById = new Map<number, [number, number]>();

  if (hasGeo) {
    for (const b of bins) {
      if (b.lat != null && b.lng != null) posById.set(b.id, [b.lng, b.lat]);
    }
    return { posById, hasGeo: true };
  }

  for (let i = 0; i < bins.length; i++) {
    const angle = (2 * Math.PI * i) / Math.max(bins.length, 1);
    posById.set(bins[i].id, [Math.cos(angle), Math.sin(angle)]);
  }
  return { posById, hasGeo: false };
}
