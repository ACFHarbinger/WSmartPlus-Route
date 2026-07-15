/** Shared log-scale heuristics for Studio ECharts panels (§G.1 / §G.7). */

import { symexp, symlog } from "./symlog";

export function isOverflowMetric(key: string): boolean {
  return /^(mean_)?overflows?$/i.test(key) || /overflow/i.test(key);
}

export function isLogScaleMetric(key: string): boolean {
  return (
    isOverflowMetric(key) ||
    /loss|cost|gap|profit|objective|reward|km|kg|entropy|grad|duration|seconds|time|count|histogram|instances/i.test(
      key
    )
  );
}

export function chartMetricDisplay(
  value: number | null | undefined,
  metricKey: string,
  logScale: boolean
): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  if (!logScale || !isLogScaleMetric(metricKey)) return value;
  if (isOverflowMetric(metricKey)) return symlog(value);
  return Math.max(value, 1e-8);
}

export function chartMetricUsesSymlog(metricKey: string, logScale: boolean): boolean {
  return logScale && isOverflowMetric(metricKey);
}

export function chartMetricYAxisType(
  metricKey: string,
  logScale: boolean
): "log" | "value" {
  if (!logScale || !isLogScaleMetric(metricKey) || isOverflowMetric(metricKey)) {
    return "value";
  }
  return "log";
}

export function displayBarValue(value: number, yKey: string, logScale: boolean): number {
  if (!logScale || !isLogScaleMetric(yKey)) return value;
  if (isOverflowMetric(yKey)) return symlog(value);
  return Math.max(value, 1e-8);
}

/** Radar / parallel-axis display value when global log-scale is on. */
export function radarAxisValue(value: number, metricKey: string, logScale: boolean): number {
  if (!logScale || !isLogScaleMetric(metricKey)) return value;
  return chartMetricDisplay(value, metricKey, true) ?? value;
}

const PARALLEL_AXIS_METRICS: Record<string, string> = {
  Overflows: "overflows",
  "kg/km": "kg/km",
  km: "km",
  Profit: "profit",
};

/** Parallel-coordinates axis value transform (§G.1.4 / §G.7). */
export function parallelAxisValue(
  value: number,
  axisName: string,
  logScale: boolean
): number {
  const metricKey = PARALLEL_AXIS_METRICS[axisName];
  if (!metricKey) return value;
  return radarAxisValue(value, metricKey, logScale);
}

/** Invert a brushed parallel-axis coordinate back to raw metric space. */
export function invertParallelAxisValue(
  value: number,
  axisName: string,
  logScale: boolean
): number {
  if (!logScale) return value;
  const metricKey = PARALLEL_AXIS_METRICS[axisName];
  if (!metricKey || !isLogScaleMetric(metricKey)) return value;
  if (isOverflowMetric(metricKey)) return symexp(value);
  return value;
}
