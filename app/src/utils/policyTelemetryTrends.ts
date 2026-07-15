/**
 * Cross-run policy telemetry chart builders (§A.3 Option C).
 */
import type { EChartsOption } from "echarts";
import type { PolicyTelemetryTrendRow, PolicyTrajectorySeries } from "../types";
import { ema, policyVizTypeLabel } from "./policyTelemetry";
import { barOpacity, isHighlighted } from "./chartHighlight";
import { downloadCsv } from "./tableExport";

export interface TrendBrushFilter {
  policy: string | null;
  runLabel: string | null;
}

/** Derive SQLite ``run_label`` from a log path (matches Python ``Path.stem``). */
export function runLabelFromPath(path: string): string {
  const base = path.split(/[/\\]/).pop() ?? path;
  const stem = base.replace(/\.[^.]+$/, "");
  return stem || base;
}

/** Scan process stdout for the first ``.jsonl`` path (newest line wins). */
export function extractJsonlPathFromLogLines(lines: string[]): string | null {
  for (let i = lines.length - 1; i >= 0; i--) {
    const match = lines[i]!.match(/([^\s'"]+\.jsonl)/);
    if (match?.[1]) return match[1];
  }
  return null;
}

/** Derive SQLite ``run_label`` from process stdout, falling back to process id. */
export function runLabelFromLogLines(lines: string[], fallbackId: string): string {
  const jsonl = extractJsonlPathFromLogLines(lines);
  if (jsonl) return runLabelFromPath(jsonl);
  return runLabelFromPath(fallbackId);
}

function trendRowHighlighted(row: PolicyTelemetryTrendRow, brush: TrendBrushFilter): boolean {
  const policyOk = isHighlighted(row.policy, brush.policy ? [brush.policy] : null);
  const runOk = !brush.runLabel || trendRowRunKey(row) === brush.runLabel;
  return policyOk && runOk;
}

function trajectoryHighlighted(series: PolicyTrajectorySeries, brush: TrendBrushFilter): boolean {
  const policyOk = isHighlighted(series.policy, brush.policy ? [brush.policy] : null);
  const runOk = !brush.runLabel || trajectoryRunKey(series) === brush.runLabel;
  return policyOk && runOk;
}

const CHART_COLORS = ["#60a5fa", "#fbbf24", "#4ade80", "#f87171", "#c084fc", "#94a3b8"];

export function trendRowLabel(row: PolicyTelemetryTrendRow): string {
  const run = row.run_label ?? row.log_path.split("/").pop() ?? "run";
  return `${run} · d${row.day}`;
}

export function trendRowRunKey(row: PolicyTelemetryTrendRow): string {
  return row.run_label ?? row.log_path;
}

/** Apply global policy / run_label brush to trend rows (§A.3 Option C). */
export function filterTrendRows(
  rows: PolicyTelemetryTrendRow[],
  policy: string | null,
  runLabel: string | null
): PolicyTelemetryTrendRow[] {
  if (!policy && !runLabel) return rows;
  return rows.filter((row) => {
    if (policy && row.policy !== policy) return false;
    if (runLabel && trendRowRunKey(row) !== runLabel) return false;
    return true;
  });
}

export function trajectoryRunKey(series: PolicyTrajectorySeries): string {
  return series.run_label ?? series.label.split(" · ")[0] ?? series.label;
}

/** Apply global policy / run_label brush to trajectory series (§A.3 Option C). */
export function filterTrajectorySeries(
  series: PolicyTrajectorySeries[],
  policy: string | null,
  runLabel: string | null
): PolicyTrajectorySeries[] {
  if (!policy && !runLabel) return series;
  return series.filter((item) => {
    if (policy && item.policy !== policy) return false;
    if (runLabel && trajectoryRunKey(item) !== runLabel) return false;
    return true;
  });
}

export function buildTrendComparisonOption(
  rows: PolicyTelemetryTrendRow[],
  theme: "dark" | "light",
  logScale = false,
  brush: TrendBrushFilter = { policy: null, runLabel: null }
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
    series: seriesKeys.map((key, idx) => {
      const sampleRow = withMetric.find((row) => trendRowLabel(row) === key);
      const runKey = sampleRow ? trendRowRunKey(sampleRow) : key;
      const runHighlighted = !brush.runLabel || runKey === brush.runLabel;
      const baseColor = CHART_COLORS[idx % CHART_COLORS.length];
      return {
        name: key,
        type: "bar" as const,
        data: policies.map((policy) => {
          const value = lookup.get(`${key}::${policy}`) ?? null;
          if (value == null) return null;
          const opacity = runHighlighted
            ? barOpacity(policy, brush.policy ? [brush.policy] : null)
            : 0.25;
          return {
            value,
            itemStyle: { color: baseColor, opacity },
          };
        }),
        itemStyle: { color: baseColor },
      };
    }),
  };
}

export function buildTrendStepsOption(
  rows: PolicyTelemetryTrendRow[],
  theme: "dark" | "light",
  brush: TrendBrushFilter = { policy: null, runLabel: null }
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
        data: rows.map((r) => ({
          value: r.step_count,
          itemStyle: {
            color: CHART_COLORS[0],
            opacity: trendRowHighlighted(r, brush) ? 1 : 0.25,
          },
        })),
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

/** Union solver step indices (iteration / generation) across trajectory series. */
export function unionTrajectoryX(series: PolicyTrajectorySeries[]): number[] {
  const seen = new Set<number>();
  for (const item of series) {
    for (const x of item.x) seen.add(x);
  }
  return [...seen].sort((a, b) => a - b);
}

export function buildTrendTrajectoryOption(
  series: PolicyTrajectorySeries[],
  theme: "dark" | "light",
  logScale = false,
  smooth = false,
  brush: TrendBrushFilter = { policy: null, runLabel: null }
): EChartsOption | null {
  if (series.length === 0) return null;

  const metricName = series[0]?.metric_name ?? "metric";
  const xValues = unionTrajectoryX(series);
  const xLabels = xValues.map(String);

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
      name: "solver step",
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
      const lookup = new Map(item.x.map((x, i) => [x, item.y[i]]));
      const aligned: Array<number | null> = xValues.map((x) => lookup.get(x) ?? null);
      let data: Array<number | null> = aligned;
      if (smooth) {
        const defined = aligned.filter((v): v is number => v != null);
        const smoothed = ema(defined);
        let j = 0;
        data = aligned.map((v) => (v == null ? null : smoothed[j++]!));
      }
      const color = CHART_COLORS[idx % CHART_COLORS.length];
      const opacity = trajectoryHighlighted(item, brush) ? 1 : 0.25;
      return {
        name: item.label,
        type: "line" as const,
        smooth,
        showSymbol: false,
        data,
        lineStyle: { width: 2, color, opacity },
        itemStyle: { color, opacity },
      };
    }),
  };
}

export function exportPolicyTelemetryTrendsCsv(rows: PolicyTelemetryTrendRow[]): void {
  downloadCsv(
    "policy-telemetry-trends.csv",
    [
      "run_label",
      "log_path",
      "policy",
      "policy_type",
      "day",
      "sample_idx",
      "step_count",
      "final_metric",
      "metric_name",
      "emitted_at",
    ],
    rows.map((row) => [
      row.run_label ?? "",
      row.log_path,
      row.policy,
      row.policy_type,
      row.day,
      row.sample_idx,
      row.step_count,
      row.final_metric ?? "",
      row.metric_name ?? "",
      row.emitted_at,
    ])
  );
}

export function exportPolicyTrajectoryCsv(series: PolicyTrajectorySeries[]): void {
  const header = ["label", "run_label", "policy", "policy_type", "day", "metric_name", "step", "value"];
  const rows: Array<Array<string | number>> = [];
  for (const item of series) {
    for (let i = 0; i < item.x.length; i++) {
      rows.push([
        item.label,
        item.run_label ?? "",
        item.policy,
        item.policy_type,
        item.day,
        item.metric_name,
        item.x[i]!,
        item.y[i]!,
      ]);
    }
  }
  downloadCsv("policy-telemetry-trajectories.csv", header, rows);
}

export { policyVizTypeLabel };
