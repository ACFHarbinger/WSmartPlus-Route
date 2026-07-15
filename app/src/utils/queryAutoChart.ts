/**
 * Suggest ECharts visualizations from DuckDB query results (§G.6).
 */

import {
  chartMetricYAxisType,
  displayBarValue,
  isLogScaleMetric,
  isOverflowMetric,
} from "./chartLogScale";
import { paretoFront, paretoStepLine } from "./pareto";

export type AutoChartType = "bar" | "grouped-bar" | "heatmap" | "line" | "scatter";

export interface AutoChartSpec {
  type: AutoChartType;
  label: string;
  xKey: string;
  yKey: string;
  /** Second dimension for grouped bars (e.g. policy within city_scale). */
  seriesKey?: string;
  /** Point label for scatter cross-filter (e.g. policy on profit vs overflows). */
  labelKey?: string;
}

const PREFERRED_DIMS = ["city_scale", "run_label", "policy", "day", "sample_id"];
const PREFERRED_METRICS = [
  "mean_kgkm",
  "kg_per_km",
  "mean_profit",
  "profit",
  "mean_overflows",
  "overflows",
  "mean_kg",
  "kg",
  "km",
];

function isNumericCol(col: string, rows: Record<string, unknown>[]): boolean {
  return rows.every((r) => {
    const v = r[col];
    if (v == null || v === "") return true;
    return !Number.isNaN(Number(v));
  });
}

function findPreferredDim(columns: string[], stringCols: string[]): string | undefined {
  const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
  for (const dim of PREFERRED_DIMS) {
    const match = lower.get(dim);
    if (match && stringCols.includes(match)) return match;
  }
  return stringCols[0];
}

function findPreferredMetric(numericCols: string[], xKey: string): string {
  const lower = new Map(numericCols.map((c) => [c.toLowerCase(), c]));
  for (const metric of PREFERRED_METRICS) {
    const match = lower.get(metric);
    if (match && match !== xKey) return match;
  }
  return numericCols.find((c) => c !== xKey) ?? numericCols[0];
}

/** All applicable chart specs for a result set (primary suggestion first). */
export function suggestChartAlternatives(
  columns: string[],
  rows: Record<string, unknown>[]
): AutoChartSpec[] {
  const primary = suggestChart(columns, rows);
  if (!primary) return [];

  const alternatives: AutoChartSpec[] = [primary];
  const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
  const numericCols = columns.filter((c) => isNumericCol(c, rows));
  const stringCols = columns.filter((c) => !numericCols.includes(c));

  if (stringCols.length >= 2 && numericCols.length >= 1) {
    const rowDim = lower.get("city_scale") ?? lower.get("run_label");
    const colDim = lower.get("policy");
    const yKey = findPreferredMetric(numericCols, rowDim ?? stringCols[0]);

    if (rowDim && colDim && rowDim !== colDim) {
      const heatmap: AutoChartSpec = {
        type: "heatmap",
        label: `${yKey} matrix (${rowDim} × ${colDim})`,
        xKey: colDim,
        seriesKey: rowDim,
        yKey,
      };
      const grouped: AutoChartSpec = {
        type: "grouped-bar",
        label: `${yKey} by ${rowDim} × ${colDim}`,
        xKey: rowDim,
        seriesKey: colDim,
        yKey,
      };
      for (const spec of [heatmap, grouped]) {
        if (!alternatives.some((a) => a.type === spec.type)) alternatives.push(spec);
      }
    }
  }

  if (stringCols.length >= 1 && numericCols.length >= 1) {
    const xKey = findPreferredDim(columns, stringCols) ?? stringCols[0];
    const yKey = findPreferredMetric(numericCols, xKey);
    const bar: AutoChartSpec = { type: "bar", label: `${yKey} by ${xKey}`, xKey, yKey };
    if (!alternatives.some((a) => a.type === "bar")) alternatives.push(bar);
  }

  const labeledScatter = findLabeledMetricScatter(columns, rows);
  if (labeledScatter && !alternatives.some((a) => a.type === "scatter")) {
    alternatives.push(labeledScatter);
  }

  const timeCol = columns.find((c) => /^(day|epoch|step|time|sample)/i.test(c));
  if (timeCol && numericCols.some((c) => c !== timeCol)) {
    const yKey = findPreferredMetric(numericCols, timeCol);
    const line: AutoChartSpec = {
      type: "line",
      label: `${yKey} over ${timeCol}`,
      xKey: timeCol,
      yKey,
    };
    if (!alternatives.some((a) => a.type === "line")) alternatives.push(line);
  }

  return alternatives;
}

function findLabeledMetricScatter(
  columns: string[],
  rows: Record<string, unknown>[]
): AutoChartSpec | null {
  const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
  const labelKey =
    lower.get("policy") ??
    lower.get("run_label") ??
    lower.get("city_scale");
  const profitKey = lower.get("mean_profit") ?? lower.get("profit");
  const overflowKey = lower.get("mean_overflows") ?? lower.get("overflows");
  if (!labelKey || !profitKey || !overflowKey) return null;
  if (!rows.every((r) => r[labelKey] != null && r[labelKey] !== "")) return null;
  return {
    type: "scatter",
    label: `${overflowKey} vs ${profitKey}`,
    xKey: profitKey,
    yKey: overflowKey,
    labelKey,
  };
}

export function suggestChart(
  columns: string[],
  rows: Record<string, unknown>[]
): AutoChartSpec | null {
  if (rows.length < 2 || columns.length < 2) return null;

  const numericCols = columns.filter((c) => isNumericCol(c, rows));
  const stringCols = columns.filter((c) => !numericCols.includes(c));

  const labeledScatter = findLabeledMetricScatter(columns, rows);
  if (labeledScatter) return labeledScatter;

  const timeCol = columns.find((c) => /^(day|epoch|step|time|sample)/i.test(c));

  if (timeCol && numericCols.some((c) => c !== timeCol)) {
    const yKey = findPreferredMetric(numericCols, timeCol);
    return { type: "line", label: `${yKey} over ${timeCol}`, xKey: timeCol, yKey };
  }

  if (stringCols.length >= 2 && numericCols.length >= 1) {
    const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
    const rowDim = lower.get("city_scale") ?? lower.get("run_label");
    const colDim = lower.get("policy");
    if (rowDim && colDim && rowDim !== colDim) {
      const yKey = findPreferredMetric(numericCols, rowDim);
      return {
        type: "heatmap",
        label: `${yKey} matrix (${rowDim} × ${colDim})`,
        xKey: colDim,
        seriesKey: rowDim,
        yKey,
      };
    }

    const groupKey =
      lower.get("city_scale") ?? lower.get("run_label") ?? findPreferredDim(columns, stringCols);
    const seriesKey =
      lower.get("policy") ?? stringCols.find((c) => c !== groupKey);
    if (groupKey && seriesKey && groupKey !== seriesKey) {
      const yKey = findPreferredMetric(numericCols, groupKey);
      return {
        type: "grouped-bar",
        label: `${yKey} by ${groupKey} × ${seriesKey}`,
        xKey: groupKey,
        seriesKey,
        yKey,
      };
    }
  }

  if (stringCols.length >= 1 && numericCols.length >= 1) {
    const xKey = findPreferredDim(columns, stringCols) ?? stringCols[0];
    const yKey = findPreferredMetric(numericCols, xKey);
    return { type: "bar", label: `${yKey} by ${xKey}`, xKey, yKey };
  }

  if (numericCols.length >= 2) {
    const xKey = findPreferredMetric(numericCols, numericCols[1]);
    const yKey = numericCols.find((c) => c !== xKey) ?? numericCols[1];
    return {
      type: "scatter",
      label: `${yKey} vs ${xKey}`,
      xKey,
      yKey,
    };
  }

  return null;
}

/** Resolve heatmap cell indices to dimension labels for cross-filter clicks. */
export function heatmapCellLabels(
  spec: AutoChartSpec,
  rows: Record<string, unknown>[],
  xIndex: number,
  yIndex: number
): { colLabel?: string; rowLabel?: string } {
  if (!spec.seriesKey) return {};
  const colLabels = [...new Set(rows.map((r) => String(r[spec.xKey] ?? "")))];
  const rowLabels = [...new Set(rows.map((r) => String(r[spec.seriesKey!] ?? "")))];
  return {
    colLabel: colLabels[xIndex],
    rowLabel: rowLabels[yIndex],
  };
}

export interface AutoChartBuildOptions {
  /** Log-scale overflows axis on profit vs overflows scatter (§G.1 / §G.6). */
  logScale?: boolean;
}

export function buildAutoChartOption(
  spec: AutoChartSpec,
  rows: Record<string, unknown>[],
  opts: AutoChartBuildOptions = {}
): Record<string, unknown> {
  const { logScale = false } = opts;
  const toNum = (v: unknown) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };
  const displayOverflow = (y: number) => (logScale ? Math.max(y, 0.001) : y);
  const displayScatterX = (x: number) =>
    logScale && isLogScaleMetric(spec.xKey) ? Math.max(x, 0.001) : x;

  if (spec.type === "bar") {
    const labels = rows.map((r) => String(r[spec.xKey] ?? ""));
    const barLog = logScale && isLogScaleMetric(spec.yKey);
    const values = rows.map((r) =>
      displayBarValue(toNum(r[spec.yKey]), spec.yKey, logScale)
    );
    return {
      backgroundColor: "transparent",
      grid: { left: 48, right: 12, top: 24, bottom: 48 },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: {
        type: chartMetricYAxisType(spec.yKey, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
        name: barLog ? spec.yKey : undefined,
      },
      series: [{ type: "bar", data: values, itemStyle: { color: "#6366f1" } }],
      tooltip: { trigger: "axis" },
    };
  }

  if (spec.type === "heatmap" && spec.seriesKey) {
    const rowLabels = [...new Set(rows.map((r) => String(r[spec.seriesKey!] ?? "")))];
    const colLabels = [...new Set(rows.map((r) => String(r[spec.xKey] ?? "")))];
    const lookup = new Map<string, number>();
    for (const row of rows) {
      lookup.set(
        `${row[spec.seriesKey]}::${row[spec.xKey]}`,
        toNum(row[spec.yKey])
      );
    }
    const flat = rowLabels.flatMap((row, ri) =>
      colLabels.map((col, ci) => [ci, ri, lookup.get(`${row}::${col}`) ?? 0] as [number, number, number])
    );
    const vals = flat.map(([, , v]) => v);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 16, top: 24, bottom: 56 },
      xAxis: {
        type: "category",
        data: colLabels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: {
        type: "category",
        data: rowLabels,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: {
        min,
        max,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
        textStyle: { color: "#9090b0", fontSize: 9 },
        show: colLabels.length > 1,
      },
      series: [
        {
          type: "heatmap",
          data: flat,
          label: {
            show: rowLabels.length * colLabels.length <= 48,
            color: "#c0c0d8",
            fontSize: 8,
          },
        },
      ],
      tooltip: { position: "top" },
    };
  }

  if (spec.type === "grouped-bar" && spec.seriesKey) {
    const groupLabels = [...new Set(rows.map((r) => String(r[spec.xKey] ?? "")))];
    const seriesLabels = [...new Set(rows.map((r) => String(r[spec.seriesKey!] ?? "")))];
    const palette = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];
    const lookup = new Map<string, number>();
    for (const row of rows) {
      lookup.set(
        `${row[spec.xKey]}::${row[spec.seriesKey]}`,
        displayBarValue(toNum(row[spec.yKey]), spec.yKey, logScale)
      );
    }
    return {
      backgroundColor: "transparent",
      grid: { left: 48, right: 12, top: 36, bottom: 48 },
      legend: { data: seriesLabels, textStyle: { color: "#9090b0", fontSize: 9 } },
      xAxis: {
        type: "category",
        data: groupLabels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: {
        type: chartMetricYAxisType(spec.yKey, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: seriesLabels.map((name, i) => ({
        name,
        type: "bar",
        data: groupLabels.map(
          (group) => lookup.get(`${group}::${name}`) ?? 0
        ),
        itemStyle: { color: palette[i % palette.length] },
      })),
      tooltip: { trigger: "axis" },
    };
  }

  if (spec.type === "line") {
    const points = rows
      .map(
        (r) =>
          [
            toNum(r[spec.xKey]),
            displayBarValue(toNum(r[spec.yKey]), spec.yKey, logScale),
          ] as [number, number]
      )
      .sort((a, b) => a[0] - b[0]);
    return {
      backgroundColor: "transparent",
      grid: { left: 48, right: 12, top: 24, bottom: 32 },
      xAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
      yAxis: {
        type: chartMetricYAxisType(spec.yKey, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "line",
          data: points,
          smooth: true,
          lineStyle: { color: "#34d399", width: 2 },
          symbolSize: 5,
        },
      ],
      tooltip: { trigger: "axis" },
    };
  }

  const scatterYLog = logScale && isOverflowMetric(spec.yKey);
  const scatterXLog = logScale && isLogScaleMetric(spec.xKey) && !isOverflowMetric(spec.xKey);
  const paretoPoints = spec.labelKey
    ? rows.map((r) => ({
        id: String(r[spec.labelKey!] ?? ""),
        x: toNum(r[spec.xKey]),
        y: toNum(r[spec.yKey]),
      }))
    : [];
  const frontIds = new Set(
    spec.labelKey ? paretoFront(paretoPoints).map((p) => p.id) : []
  );
  const step = spec.labelKey ? paretoStepLine(paretoFront(paretoPoints)) : [];

  const points = spec.labelKey
    ? rows.map((r) => {
        const name = String(r[spec.labelKey!] ?? "");
        const y = toNum(r[spec.yKey]);
        const onFront = frontIds.has(name);
        return {
          name,
          value: [displayScatterX(toNum(r[spec.xKey])), displayOverflow(y)] as [number, number],
          itemStyle: { color: onFront ? "#34d399" : "#6366f1" },
        };
      })
    : rows.map((r) => [toNum(r[spec.xKey]), toNum(r[spec.yKey])]);

  const series: Record<string, unknown>[] = [
    {
      type: "scatter",
      name: "Policies",
      data: points,
      symbolSize: spec.labelKey ? 10 : 8,
      itemStyle: spec.labelKey ? undefined : { color: "#6366f1" },
      label: spec.labelKey
        ? { show: rows.length <= 24, position: "top", color: "#9090b0", fontSize: 8 }
        : undefined,
    },
  ];

  if (step.length > 1) {
    series.push({
      type: "line",
      name: "Pareto front",
      data: step.map(([x, y]) => [displayScatterX(x), displayOverflow(y)]),
      lineStyle: { color: "#f3f4f6", type: "dashed", width: 1 },
      symbol: "none",
      tooltip: { show: false },
      z: 1,
    });
  }

  return {
    backgroundColor: "transparent",
    grid: { left: 48, right: 12, top: 24, bottom: 32 },
    legend:
      step.length > 1
        ? { data: ["Policies", "Pareto front"], textStyle: { color: "#9090b0", fontSize: 9 } }
        : undefined,
    xAxis: {
      type: scatterXLog ? "log" : "value",
      logBase: 10,
      name: spec.xKey,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    yAxis: {
      type: scatterYLog ? "log" : "value",
      logBase: 10,
      name: spec.yKey,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    series,
    tooltip: { trigger: "item" },
  };
}
