/**
 * Suggest ECharts visualizations from DuckDB query results (§G.6).
 */

export type AutoChartType = "bar" | "grouped-bar" | "line" | "scatter";

export interface AutoChartSpec {
  type: AutoChartType;
  label: string;
  xKey: string;
  yKey: string;
  /** Second dimension for grouped bars (e.g. policy within city_scale). */
  seriesKey?: string;
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

export function suggestChart(
  columns: string[],
  rows: Record<string, unknown>[]
): AutoChartSpec | null {
  if (rows.length < 2 || columns.length < 2) return null;

  const numericCols = columns.filter((c) => isNumericCol(c, rows));
  const stringCols = columns.filter((c) => !numericCols.includes(c));

  const timeCol = columns.find((c) => /^(day|epoch|step|time|sample)/i.test(c));

  if (timeCol && numericCols.some((c) => c !== timeCol)) {
    const yKey = findPreferredMetric(numericCols, timeCol);
    return { type: "line", label: `${yKey} over ${timeCol}`, xKey: timeCol, yKey };
  }

  if (stringCols.length >= 2 && numericCols.length >= 1) {
    const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
    const groupKey =
      lower.get("city_scale") ?? lower.get("run_label") ?? findPreferredDim(columns, stringCols);
    const seriesKey =
      lower.get("policy") ??
      stringCols.find((c) => c !== groupKey);
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

export function buildAutoChartOption(
  spec: AutoChartSpec,
  rows: Record<string, unknown>[]
): Record<string, unknown> {
  const toNum = (v: unknown) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  };

  if (spec.type === "bar") {
    const labels = rows.map((r) => String(r[spec.xKey] ?? ""));
    const values = rows.map((r) => toNum(r[spec.yKey]));
    return {
      backgroundColor: "transparent",
      grid: { left: 48, right: 12, top: 24, bottom: 48 },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
      series: [{ type: "bar", data: values, itemStyle: { color: "#6366f1" } }],
      tooltip: { trigger: "axis" },
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
        toNum(row[spec.yKey])
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
      yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
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
      .map((r) => [toNum(r[spec.xKey]), toNum(r[spec.yKey])] as [number, number])
      .sort((a, b) => a[0] - b[0]);
    return {
      backgroundColor: "transparent",
      grid: { left: 48, right: 12, top: 24, bottom: 32 },
      xAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
      yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
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

  const points = rows.map((r) => [toNum(r[spec.xKey]), toNum(r[spec.yKey])]);
  return {
    backgroundColor: "transparent",
    grid: { left: 48, right: 12, top: 24, bottom: 32 },
    xAxis: { type: "value", name: spec.xKey, axisLabel: { color: "#9090b0", fontSize: 9 } },
    yAxis: { type: "value", name: spec.yKey, axisLabel: { color: "#9090b0", fontSize: 9 } },
    series: [
      {
        type: "scatter",
        data: points,
        symbolSize: 8,
        itemStyle: { color: "#6366f1" },
      },
    ],
    tooltip: { trigger: "item" },
  };
}
