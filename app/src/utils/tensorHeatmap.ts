/**
 * ECharts heatmap builders for tensor slices and loss landscapes (§G.5).
 */

export function buildMatrixHeatmapOption(
  values: number[][],
  opts: {
    title?: string;
    min?: number;
    max?: number;
    theme?: "dark" | "light";
    xLabel?: string;
    yLabel?: string;
  } = {}
): Record<string, unknown> {
  const { title, min, max, theme = "dark", xLabel = "Col", yLabel = "Row" } = opts;
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const flat = values.flat();
  const dataMin = min ?? Math.min(...flat);
  const dataMax = max ?? Math.max(...flat);

  const data: Array<[number, number, number | string]> = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r][c];
      data.push([c, r, Number.isFinite(v) ? v : "-"]);
    }
  }

  return {
    backgroundColor: "transparent",
    title: title
      ? {
          text: title,
          left: "center",
          top: 4,
          textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" },
        }
      : undefined,
    tooltip: {
      position: "top",
      formatter: (p: { value?: [number, number, number] }) => {
        const v = p.value;
        if (!v) return "";
        return `${yLabel} ${v[1]} · ${xLabel} ${v[0]}<br/>${Number(v[2]).toFixed(4)}`;
      },
    },
    grid: { left: 48, right: 56, top: title ? 36 : 16, bottom: 36 },
    xAxis: {
      type: "category",
      name: xLabel,
      data: Array.from({ length: cols }, (_, i) => String(i)),
      splitArea: { show: true },
      axisLabel: { color: "#9090b0", fontSize: 9, interval: Math.max(0, Math.floor(cols / 12)) },
    },
    yAxis: {
      type: "category",
      name: yLabel,
      data: Array.from({ length: rows }, (_, i) => String(i)),
      splitArea: { show: true },
      axisLabel: { color: "#9090b0", fontSize: 9, interval: Math.max(0, Math.floor(rows / 12)) },
    },
    visualMap: {
      min: dataMin,
      max: dataMax,
      calculable: true,
      orient: "vertical",
      right: 4,
      top: "center",
      inRange: {
        color: ["#1e3a8a", "#6366f1", "#fbbf24", "#ef4444"],
      },
      textStyle: { color: "#9090b0", fontSize: 9 },
    },
    series: [
      {
        type: "heatmap",
        data,
        emphasis: { itemStyle: { shadowBlur: 6, shadowColor: "rgba(0,0,0,0.4)" } },
      },
    ],
  };
}

/** Pick arrays that look like attention weight matrices. */
export function suggestAttentionKeys(arrays: { key: string; shape: number[] }[]): string[] {
  return arrays
    .filter((a) => {
      const rank = a.shape.length;
      if (rank < 2) return false;
      const last = a.shape[rank - 1];
      const prev = a.shape[rank - 2];
      return last >= 4 && prev >= 4 && /attn|attention|weights|alpha/i.test(a.key);
    })
    .map((a) => a.key);
}

export function leadingIndexCount(shape: number[]): number {
  return Math.max(0, shape.length - 2);
}

export function defaultIndices(shape: number[]): number[] {
  return Array.from({ length: leadingIndexCount(shape) }, () => 0);
}
