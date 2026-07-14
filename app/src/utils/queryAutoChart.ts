/**
 * Suggest ECharts visualizations from DuckDB query results (§G.6).
 */

export type AutoChartType = "bar" | "line" | "scatter";

export interface AutoChartSpec {
  type: AutoChartType;
  label: string;
  xKey: string;
  yKey: string;
}

function isNumericCol(col: string, rows: Record<string, unknown>[]): boolean {
  return rows.every((r) => {
    const v = r[col];
    if (v == null || v === "") return true;
    return !Number.isNaN(Number(v));
  });
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
    const yKey = numericCols.find((c) => c !== timeCol) ?? numericCols[0];
    return { type: "line", label: `${yKey} over ${timeCol}`, xKey: timeCol, yKey };
  }

  if (stringCols.length >= 1 && numericCols.length >= 1) {
    const xKey = stringCols[0];
    const yKey = numericCols.find((c) => c !== xKey) ?? numericCols[0];
    return { type: "bar", label: `${yKey} by ${xKey}`, xKey, yKey };
  }

  if (numericCols.length >= 2) {
    return {
      type: "scatter",
      label: `${numericCols[1]} vs ${numericCols[0]}`,
      xKey: numericCols[0],
      yKey: numericCols[1],
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
