/**
 * Failure route-diff overlay helpers (§A.6 Option C).
 *
 * Shared by ECharts ``RouteViz`` and deck.gl ``DeckRouteMap`` to highlight
 * overflow bins (red) and skipped high-fill bins (orange), plus optional
 * tour-diff highlights when comparing two policies.
 */

import type { SimDayData, SimFailureSummary } from "../../types";

export type FailureBinKind = "overflow" | "skipped";

export interface FailureBinIdSets {
  overflowIds: Set<number>;
  skippedIds: Set<number>;
}

export interface TourDiffSets {
  onlyFirst: Set<number>;
  onlySecond: Set<number>;
  shared: Set<number>;
}

export function failureBinIdSets(
  summary: SimFailureSummary | null | undefined
): FailureBinIdSets {
  const overflowIds = new Set(
    (summary?.overflow_bins ?? []).map((b) => b.bin_id)
  );
  const skippedIds = new Set(
    (summary?.skipped_high_fill_bins ?? []).map((b) => b.bin_id)
  );
  return { overflowIds, skippedIds };
}

export function resolveFailureOverlay(
  data: SimDayData,
  explicit?: SimFailureSummary | null
): SimFailureSummary | null {
  return explicit ?? data.failure_analysis ?? null;
}

export function hasFailureOverlay(summary: SimFailureSummary | null | undefined): boolean {
  if (!summary?.has_failure) return false;
  return (
    (summary.overflow_bins?.length ?? 0) > 0 ||
    (summary.skipped_high_fill_bins?.length ?? 0) > 0
  );
}

/** Tour bin ids excluding depot (-1). */
export function tourBinIds(data: SimDayData): Set<number> {
  const ids = new Set<number>();
  for (const id of data.tour_indices ?? []) {
    if (id >= 0) ids.add(id);
  }
  return ids;
}

export function computeTourDiff(first: SimDayData, second: SimDayData): TourDiffSets {
  const a = tourBinIds(first);
  const b = tourBinIds(second);
  const onlyFirst = new Set<number>();
  const onlySecond = new Set<number>();
  const shared = new Set<number>();

  for (const id of a) {
    if (b.has(id)) shared.add(id);
    else onlyFirst.add(id);
  }
  for (const id of b) {
    if (!a.has(id)) onlySecond.add(id);
  }

  return { onlyFirst, onlySecond, shared };
}

export const FAILURE_RGB: Record<FailureBinKind, [number, number, number]> = {
  overflow: [239, 68, 68],
  skipped: [251, 146, 60],
};

export const TOUR_DIFF_RGB = {
  onlyFirst: [34, 211, 238] as [number, number, number],
  onlySecond: [192, 132, 252] as [number, number, number],
};
