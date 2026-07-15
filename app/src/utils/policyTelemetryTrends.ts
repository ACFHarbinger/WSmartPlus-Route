/**
 * Cross-run policy telemetry chart builders (§A.3 Option C).
 */
import type { EChartsOption } from "echarts";
import type { PolicyTelemetryTrendRow, PolicyTrajectorySeries } from "../types";
import { ema, policyVizTypeLabel } from "./policyTelemetry";

const CHART_COLORS = ["#60a5fa", "#fbbf24", "#4ade80", "#f87171", "#c084fc", "#94a3b8"];

export function trendRowLabel(row: PolicyTelemetryTrendRow): string {
  const run = row.run_label ?? row.log_path.split("/").pop() ?? "run";
  return `${run} · d${row.day}`;
}

export function buildTrendComparisonOption(
  rows: PolicyTelemetryTrendRow[],
  theme: "dark" | "light",
  logScale = false
): EChartsOption | null {
  const withMetric = rows.filter((r) => r.final_metric != null);
  if (withMetric.length === 0) return null;

  const policies = [...new Set(withMetric.map((r) => r.policy))].sort();
  const seriesKeys = [...new Set(withMetric.map((r) => trendRowLabel(r)))];
  const metricName = withMetric[0]?.metric_name ?? "metric";

  const lookup = new Map<string, number>();
  for (const row of withMetric) {
    lookup.set(`${trendRowLabel(row)}::${row.policy}`, row.final_metric!);
  }

  return {
    backgroundColor: "transparent",
    title: {
      text: `Cross-run ${metricName}`,
      left: 8,
      top: 4,
      textStyle: { color: theme === "dark" ? "#d1d5db" : "#374151", fontSize: 12 },
    },
    grid: { left: 56, right: 16, top: 40, bottom: 56 },
    tooltip: { trigger: "axis" },
    legend: {
      type: "scroll",
      top: 4,
      right: 8,
      textStyle: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
    },
    xAxis: {
      type: "category",
      data: policies,
      axisLabel: {
        color: theme === "dark" ? "#9ca3af" : "#6b7280",
        fontSize: 9,
        rotate: 18,
        formatter: (v: string) => (v.length > 28 ? `${v.slice(0, 26)}…` : v),
      },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      name: metricName,
      nameTextStyle: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: theme === "dark" ? "#1f2937" : "#e5e7eb" } },
    },
    series: seriesKeys.map((key, idx) => ({
      name: key,
      type: "bar" as const,
      data: policies.map((policy) => lookup.get(`${key}::${policy}`) ?? null),
      itemStyle: { color: CHART_COLORS[idx % CHART_COLORS.length] },
    })),
  };
}

export function buildTrendStepsOption(
  rows: PolicyTelemetryTrendRow[],
  theme: "dark" | "light"
): EChartsOption | null {
  if (rows.length === 0) return null;

  const labels = rows.map((r) => `${r.policy.slice(0, 20)} (${trendRowLabel(r)})`);
  return {
    backgroundColor: "transparent",
    title: {
      text: "Solver steps (latest snapshots)",
      left: 8,
      top: 4,
      textStyle: { color: theme === "dark" ? "#d1d5db" : "#374151", fontSize: 12 },
    },
    grid: { left: 48, right: 16, top: 36, bottom: 72 },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: {
        color: theme === "dark" ? "#9ca3af" : "#6b7280",
        fontSize: 8,
        rotate: 28,
      },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: theme === "dark" ? "#1f2937" : "#e5e7eb" } },
    },
    series: [
      {
        type: "bar",
        data: rows.map((r) => r.step_count),
        itemStyle: { color: CHART_COLORS[0] },
      },
    ],
  };
}

export function formatTrendMetric(row: PolicyTelemetryTrendRow): string {
  if (row.final_metric == null) return "—";
  const name = row.metric_name ?? "metric";
  return `${row.final_metric.toLocaleString(undefined, { maximumFractionDigits: 3 })} (${name})`;
}

export function buildTrendTrajectoryOption(
  series: PolicyTrajectorySeries[],
  theme: "dark" | "light",
  logScale = false,
  smooth = false
): EChartsOption | null {
  if (series.length === 0) return null;

  const metricName = series[0]?.metric_name ?? "metric";
  const maxLen = Math.max(...series.map((s) => s.x.length));
  const xLabels = Array.from({ length: maxLen }, (_, i) => String(i));

  return {
    backgroundColor: "transparent",
    title: {
      text: `Improvement trajectories (${metricName})`,
      left: 8,
      top: 4,
      textStyle: { color: theme === "dark" ? "#d1d5db" : "#374151", fontSize: 12 },
    },
    grid: { left: 56, right: 16, top: 44, bottom: 40 },
    tooltip: { trigger: "axis" },
    legend: {
      type: "scroll",
      top: 4,
      right: 8,
      textStyle: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
    },
    xAxis: {
      type: "category",
      name: "step",
      data: xLabels,
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      name: metricName,
      nameTextStyle: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: theme === "dark" ? "#1f2937" : "#e5e7eb" } },
    },
    series: series.map((item, idx) => {
      const values = smooth ? ema(item.y) : item.y;
      const padded: Array<number | null> = [...values];
      while (padded.length < maxLen) padded.push(null);
      return {
        name: item.label,
        type: "line" as const,
        smooth: smooth,
        showSymbol: false,
        data: padded,
        lineStyle: { width: 2 },
        itemStyle: { color: CHART_COLORS[idx % CHART_COLORS.length] },
      };
    }),
  };
}

export { policyVizTypeLabel };
