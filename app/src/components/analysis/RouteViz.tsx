/**
 * Interactive route solution visualizer — ECharts spatial panel (§A.1 Option A).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { FailureOverlayLegend } from "./FailureOverlayLegend";
import { buildRouteVizOption } from "../../utils/routeViz";
import { hasFailureOverlay } from "../../utils/routeFailureOverlay";
import type { SimDayData, SimFailureSummary } from "../../types";

export interface RouteVizProps {
  data: SimDayData;
  title?: string;
  subtitle?: string;
  height?: number;
  filenameStem?: string;
  showExport?: boolean;
  failureOverlay?: SimFailureSummary | null;
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
  className = "",
}: RouteVizProps) {
  const chartRef = useRef<EChartsReact | null>(null);

  const option = useMemo(
    () =>
      buildRouteVizOption(data, {
        title: subtitle ? `${title} · ${subtitle}` : title,
        failureOverlay: failureOverlay ?? data.failure_analysis ?? null,
      }),
    [data, title, subtitle, failureOverlay]
  );

  if (!option) {
    return (
      <p className="text-xs text-canvas-muted py-4 text-center">
        No bin coordinates in this day log — route map unavailable.
      </p>
    );
  }

  const resolvedFailure = failureOverlay ?? data.failure_analysis ?? null;
  const showFailureLegend = hasFailureOverlay(resolvedFailure);

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
          {showExport && (
            <ChartExportButtons chartRef={chartRef} filenameStem={filenameStem} />
          )}
        </div>
      </div>
      {showFailureLegend && <FailureOverlayLegend />}
      <ReactECharts ref={chartRef} option={option} style={{ height }} />
    </div>
  );
}
