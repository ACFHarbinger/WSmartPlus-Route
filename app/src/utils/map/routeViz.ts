/**
 * ECharts route solution visualizer (§A.1 Option A / §D.1).
 *
 * Renders depot (star), customer nodes (circles sized by demand/fill), and
 * per-vehicle route edges from simulation day JSON payloads.
 */

import type { SimDayData, SimFailureSummary } from "../../types";
import {
  failureBinIdSets,
  TOUR_DIFF_RGB,
  type TourDiffSets,
} from "./routeFailureOverlay";
import { resolveBinPositions } from "./mapPositions";
import { splitVehicleTourIndices, VEHICLE_COLORS_RGB } from "./vehicleTours";

export function fillColorForPct(pct: number): string {
  if (pct >= 100) return "#f87171";
  if (pct >= 80) return "#fbbf24";
  return "#34d399";
}

/** Map fill fraction (0–1) or demand to scatter symbol size. */
export function nodeSizeFromDemand(fill: number, collected = 0, mandatory = false): number {
  const pct = Math.min(100, Math.max(0, fill * 100));
  const base = mandatory ? 11 : 8;
  const fillScale = 0.6 + (pct / 100) * 0.8;
  const collectedScale = collected > 0 ? 1 + Math.min(collected, 50) / 80 : 1;
  return base * fillScale * collectedScale;
}

export interface RouteVizBuildOptions {
  title?: string;
  failureOverlay?: SimFailureSummary | null;
  showFailureOverlay?: boolean;
  compareData?: SimDayData | null;
  tourDiff?: TourDiffSets | null;
  showTourDiff?: boolean;
  primaryLabel?: string;
  compareLabel?: string;
}

function rgbCss([r, g, b]: [number, number, number]): string {
  return `rgb(${r},${g},${b})`;
}

function nodeBorderStyle(
  binId: number,
  mandatory: boolean,
  showFailure: boolean,
  overflowIds: Set<number>,
  skippedIds: Set<number>,
  showTourDiff: boolean,
  tourDiff?: TourDiffSets | null
): { borderColor?: string; borderWidth: number } {
  if (showTourDiff && tourDiff) {
    if (tourDiff.onlyFirst.has(binId)) {
      return { borderColor: rgbCss(TOUR_DIFF_RGB.onlyFirst), borderWidth: 3 };
    }
    if (tourDiff.onlySecond.has(binId)) {
      return { borderColor: rgbCss(TOUR_DIFF_RGB.onlySecond), borderWidth: 3 };
    }
  }

  if (showFailure) {
    if (overflowIds.has(binId) || skippedIds.has(binId)) {
      return { borderColor: "#ffffff", borderWidth: 2 };
    }
  }

  if (mandatory) {
    return { borderColor: "#a78bfa", borderWidth: 2 };
  }

  return { borderWidth: 0 };
}

function nodeFillColor(
  binId: number,
  fillPct: number,
  showFailure: boolean,
  overflowIds: Set<number>,
  skippedIds: Set<number>
): string {
  if (showFailure && overflowIds.has(binId)) return "#ef4444";
  if (showFailure && skippedIds.has(binId)) return "#fb923c";
  return fillColorForPct(fillPct);
}

function buildVehiclePathSeries(
  data: SimDayData,
  posById: Map<number, [number, number]>,
  depotPos: [number, number] | undefined,
  paletteOffset: number,
  namePrefix: string
): Record<string, unknown>[] {
  const segments = splitVehicleTourIndices(data);
  return segments.map((segment, vi) => {
    const pathCoords: [number, number][] = [];
    if (depotPos) pathCoords.push(depotPos);
    for (const binId of segment) {
      const pos = posById.get(binId);
      if (pos) pathCoords.push(pos);
    }
    if (depotPos && pathCoords.length > 1) pathCoords.push(depotPos);
    const [r, g, b] = VEHICLE_COLORS_RGB[(vi + paletteOffset) % VEHICLE_COLORS_RGB.length];
    const routeName =
      segments.length > 1 ? `${namePrefix} · V${vi + 1}` : namePrefix;
    return {
      name: routeName,
      type: "line" as const,
      data: pathCoords,
      lineStyle: { color: `rgb(${r},${g},${b})`, width: 2 },
      symbol: "none",
      z: 1,
    };
  });
}

export function buildRouteVizOption(
  data: SimDayData,
  opts: RouteVizBuildOptions = {}
): Record<string, unknown> | null {
  const { all_bin_coords, tour_indices, bin_state_c, bin_state_collected, mandatory } = data;
  if (!all_bin_coords?.length) return null;

  const showFailure = opts.showFailureOverlay !== false;
  const showTourDiff = opts.showTourDiff === true && opts.tourDiff != null;
  const tourDiff = opts.tourDiff ?? null;

  const { posById } = resolveBinPositions(all_bin_coords);
  const mandatorySet = new Set(mandatory ?? []);
  const tourSet = new Set(tour_indices ?? []);

  const { overflowIds, skippedIds } = showFailure
    ? failureBinIdSets(opts.failureOverlay)
    : { overflowIds: new Set<number>(), skippedIds: new Set<number>() };

  const idleBins = all_bin_coords
    .filter((b) => b.id >= 0 && !tourSet.has(b.id))
    .map((b) => {
      const pos = posById.get(b.id);
      if (!pos) return null;
      const fill = bin_state_c?.[b.id] ?? 0;
      const isOverflow = showFailure && overflowIds.has(b.id);
      const border = nodeBorderStyle(
        b.id,
        false,
        showFailure,
        overflowIds,
        skippedIds,
        showTourDiff,
        tourDiff
      );
      return {
        value: pos,
        name: `#${b.id}`,
        symbolSize: nodeSizeFromDemand(fill),
        itemStyle: {
          color: showFailure && (overflowIds.has(b.id) || skippedIds.has(b.id))
            ? nodeFillColor(b.id, fill * 100, showFailure, overflowIds, skippedIds)
            : "#4b5563",
          ...border,
        },
        label: isOverflow
          ? { show: true, formatter: "OVF", fontSize: 8, color: "#fecaca" }
          : undefined,
      };
    })
    .filter(Boolean);

  const tourStops = (tour_indices ?? []).map((binId) => {
    const pos = posById.get(binId);
    const fill = bin_state_c?.[binId] ?? 0;
    const collected = bin_state_collected?.[binId] ?? 0;
    const pct = Math.min(100, fill * 100);
    const border = nodeBorderStyle(
      binId,
      mandatorySet.has(binId),
      showFailure,
      overflowIds,
      skippedIds,
      showTourDiff,
      tourDiff
    );
    return pos
      ? {
          value: pos,
          name: `#${binId}`,
          symbolSize: nodeSizeFromDemand(fill, collected, mandatorySet.has(binId)),
          itemStyle: {
            color: nodeFillColor(binId, pct, showFailure, overflowIds, skippedIds),
            ...border,
          },
        }
      : null;
  }).filter(Boolean);

  const depotPos = posById.get(-1);

  const primaryLabel = opts.primaryLabel ?? "Route";
  const vehiclePaths = buildVehiclePathSeries(data, posById, depotPos, 0, primaryLabel);

  const comparePaths =
    opts.compareData?.all_bin_coords?.length
      ? buildVehiclePathSeries(
          opts.compareData,
          posById,
          depotPos,
          3,
          opts.compareLabel ?? "Compare"
        )
      : [];

  return {
    backgroundColor: "transparent",
    title: opts.title
      ? { text: opts.title, left: "center", textStyle: { color: "#9090b0", fontSize: 11 } }
      : undefined,
    grid: { left: 40, right: 20, top: opts.title ? 36 : 20, bottom: 30 },
    xAxis: { type: "value", scale: true, axisLabel: { color: "#9090b0", fontSize: 10 } },
    yAxis: { type: "value", scale: true, axisLabel: { color: "#9090b0", fontSize: 10 } },
    series: [
      ...vehiclePaths,
      ...comparePaths,
      {
        name: "Idle bins",
        type: "scatter",
        data: idleBins,
        z: 2,
      },
      {
        name: "Tour stops",
        type: "scatter",
        data: tourStops,
        z: 3,
      },
      ...(depotPos
        ? [
            {
              name: "Depot",
              type: "scatter" as const,
              data: [{ value: depotPos, name: "Depot" }],
              symbol: "star",
              symbolSize: 16,
              itemStyle: { color: "#a78bfa", borderColor: "#e9d5ff", borderWidth: 1 },
              z: 4,
            },
          ]
        : []),
    ],
    tooltip: {
      trigger: "item",
      formatter: (p: { seriesName?: string; name?: string; data?: { name?: string } }) => {
        const label = p.name ?? p.data?.name ?? "";
        if (p.seriesName?.includes("· V") || p.seriesName === "Route" || p.seriesName?.includes("Compare")) {
          return `${p.seriesName}`;
        }
        return label ? `${label}${p.seriesName ? ` · ${p.seriesName}` : ""}` : p.seriesName ?? "";
      },
    },
    legend: { textStyle: { color: "#9090b0", fontSize: 9 }, top: 0 },
  };
}
