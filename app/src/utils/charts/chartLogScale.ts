/** Shared log-scale heuristics for Studio ECharts panels (§G.1 / §G.7). */

import { symexp, symlog } from "./symlog";

export function isOverflowMetric(key: string): boolean {
  return /^(mean_)?overflows?$/i.test(key) || /overflow/i.test(key);
}

export function isLogScaleMetric(key: string): boolean {
  return (
    isOverflowMetric(key) ||
    /loss|cost|gap|profit|objective|reward|km|kg|entropy|grad|duration|seconds|time|count|histogram|instances|attention|weight/i.test(
      key
    )
  );
}

/** Attention edge/opacity mapping when global log-scale is on (§G.5.3). */
export function attentionWeightDisplay(value: number, logScale: boolean): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  if (!logScale) return value;
  return chartMetricDisplay(value, "attention", true) ?? value;
}

/** ACO pheromone edge opacity/width mapping when global log-scale is on (§G.4 / §G.7). */
export function pheromoneWeightDisplay(value: number, logScale: boolean): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  if (!logScale) return value;
  return chartMetricDisplay(value, "pheromone", true) ?? value;
}

/** Transform a 2-D matrix for log-scale heatmap colour mapping; preserves raw tooltips separately. */
export function transformMatrixLogScale(
  values: number[][],
  metricKey: string,
  logScale: boolean
): number[][] {
  if (!logScale) return values;
  return values.map((row) =>
    row.map((v) => (Number.isFinite(v) ? (chartMetricDisplay(v, metricKey, true) ?? v) : v))
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

/** Horizontal offset for whiskers on grouped/category bar charts. */
export function groupedBarWhiskerX(
  api: {
    coord: (v: [number, number]) => [number, number];
    size: (v: [number, number]) => [number, number];
  },
  categoryIndex: number,
  seriesIndex: number,
  seriesCount: number,
  centerValue: number
): number {
  const bandWidth = api.size([1, 0])[0];
  const offset = (seriesIndex - (seriesCount - 1) / 2) * (bandWidth / seriesCount);
  return api.coord([categoryIndex, centerValue])[0] + offset;
}

/** Display-space mean ± std whisker endpoints for error bars (§G.1 / §G.7). */
export function errorBarBounds(
  mean: number,
  std: number,
  metricKey: string,
  logScale: boolean,
  symlogMode = false
): { low: number; high: number; center: number } {
  const rawLow = Math.max(0, mean - std);
  const rawHigh = mean + std;
  if (!logScale) {
    return { low: rawLow, high: rawHigh, center: mean };
  }
  if (symlogMode || chartMetricUsesSymlog(metricKey, true)) {
    return {
      low: symlog(rawLow),
      high: symlog(rawHigh),
      center: symlog(mean),
    };
  }
  const floor = 1e-8;
  return {
    low: Math.max(rawLow, floor),
    high: Math.max(rawHigh, floor),
    center: Math.max(mean, floor),
  };
}
