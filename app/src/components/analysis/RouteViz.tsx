/**
 * Interactive route solution visualizer — ECharts spatial panel (§A.1 Option A).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { FailureOverlayLegend } from "./FailureOverlayLegend";
import { buildRouteVizOption } from "../../utils/routeViz";
import {
  computeTourDiff,
  hasFailureOverlay,
  resolveFailureOverlay,
} from "../../utils/routeFailureOverlay";
import type { SimDayData, SimFailureSummary } from "../../types";

export interface RouteVizProps {
  data: SimDayData;
  title?: string;
  subtitle?: string;
  height?: number;
  filenameStem?: string;
  showExport?: boolean;
  failureOverlay?: SimFailureSummary | null;
  showFailureOverlay?: boolean;
  compareData?: SimDayData | null;
  compareLabel?: string;
  primaryLabel?: string;
  showTourDiff?: boolean;
  className?: string;
}

export function RouteViz({
  data,
  title = "Route Solution",
  subtitle,
  height = 280,
  filenameStem = "route-viz",
  showExport = true,
  failureOverlay,
  showFailureOverlay = true,
  compareData,
  compareLabel,
  primaryLabel,
  showTourDiff = false,
  className = "",
}: RouteVizProps) {
  const chartRef = useRef<EChartsReact | null>(null);

  const resolvedFailure = resolveFailureOverlay(data, failureOverlay);
  const tourDiff = useMemo(
    () => (showTourDiff && compareData ? computeTourDiff(data, compareData) : null),
    [showTourDiff, compareData, data]
  );

  const option = useMemo(
    () =>
      buildRouteVizOption(data, {
        title: subtitle ? `${title} · ${subtitle}` : title,
        failureOverlay: resolvedFailure,
        showFailureOverlay,
        compareData,
        compareLabel,
        primaryLabel,
        tourDiff,
        showTourDiff: showTourDiff && compareData != null,
      }),
    [
      data,
      title,
      subtitle,
      resolvedFailure,
      showFailureOverlay,
      compareData,
      compareLabel,
      primaryLabel,
      tourDiff,
      showTourDiff,
    ]
  );

  if (!option) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        No bin coordinates in this day log — route map unavailable.
      </p>
    );
  }

  const showFailureLegend =
    showFailureOverlay && hasFailureOverlay(resolvedFailure);
  const showDiffLegend =
    showTourDiff &&
    compareData != null &&
    tourDiff != null &&
    (tourDiff.onlyFirst.size > 0 || tourDiff.onlySecond.size > 0);

  return (
    <div className={`card space-y-2 ${className}`.trim()}>
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs text-canvas-muted truncate">{title}</p>
          {subtitle && (
            <p className="text-[10px] text-canvas-muted/80 font-mono truncate">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {showFailureLegend && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-danger/20 text-accent-danger">
              Failure overlay
            </span>
          )}
          {showDiffLegend && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-secondary/20 text-accent-secondary">
              Route diff
            </span>
          )}
          {showExport && (
            <ChartExportButtons chartRef={chartRef} filenameStem={filenameStem} />
          )}
        </div>
      </div>
      {(showFailureLegend || showDiffLegend) && (
        <FailureOverlayLegend
          showOverflow={showFailureLegend}
          showSkipped={showFailureLegend}
          showTourDiff={showDiffLegend}
          tourDiffLabels={
            showDiffLegend && primaryLabel && compareLabel
              ? [primaryLabel, compareLabel]
              : showDiffLegend && compareLabel
                ? ["Policy A", compareLabel]
                : undefined
          }
        />
      )}
      <ReactECharts ref={chartRef} option={option} style={{ height }} />
    </div>
  );
}
