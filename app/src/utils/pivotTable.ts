/**
 * Client-side pivot aggregation for DuckDB/CSV result grids (§G.6).
 */

export type PivotAgg = "mean" | "sum" | "count";

export interface PivotConfig {
  rowKey: string;
  colKey: string | null;
  valueKey: string;
  agg: PivotAgg;
}

export interface PivotResult {
  rowLabels: string[];
  colLabels: string[];
  cells: number[][];
}

function toNum(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function aggregate(vals: number[], agg: PivotAgg): number {
  if (vals.length === 0) return 0;
  if (agg === "count") return vals.length;
  if (agg === "sum") return vals.reduce((a, b) => a + b, 0);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export function buildPivot(
  rows: Record<string, unknown>[],
  config: PivotConfig
): PivotResult | null {
  const { rowKey, colKey, valueKey, agg } = config;
  if (!rows.length || !rowKey || !valueKey) return null;

  const rowSet = new Set<string>();
  const colSet = new Set<string>();
  const bucket = new Map<string, number[]>();

  for (const row of rows) {
    const rk = String(row[rowKey] ?? "—");
    const ck = colKey ? String(row[colKey] ?? "—") : "_value";
    const num = toNum(row[valueKey]);
    rowSet.add(rk);
    colSet.add(ck);
    const key = `${rk}\0${ck}`;
    const list = bucket.get(key) ?? [];
    if (num != null) list.push(num);
    else if (agg === "count") list.push(0);
    bucket.set(key, list);
  }

  const rowLabels = [...rowSet].sort();
  const colLabels = colKey ? [...colSet].sort() : ["Total"];

  const cells = rowLabels.map((rl) =>
    colLabels.map((cl) => {
      const vals = bucket.get(`${rl}\0${cl}`) ?? [];
      return aggregate(vals, agg);
    })
  );

  return { rowLabels, colLabels, cells };
}

export function pivotHeatmapOption(result: PivotResult, title: string) {
  const flat: Array<[number, number, number]> = [];
  let min = Infinity;
  let max = -Infinity;
  for (let ri = 0; ri < result.rowLabels.length; ri++) {
    for (let ci = 0; ci < result.colLabels.length; ci++) {
      const v = result.cells[ri][ci];
      min = Math.min(min, v);
      max = Math.max(max, v);
      flat.push([ci, ri, v]);
    }
  }
  return {
    backgroundColor: "transparent",
    title: { text: title, left: "center", textStyle: { color: "#9090b0", fontSize: 10 } },
    grid: { left: 72, right: 16, top: 36, bottom: 48 },
    xAxis: {
      type: "category",
      data: result.colLabels,
      axisLabel: { color: "#9090b0", fontSize: 8, rotate: 20 },
    },
    yAxis: {
      type: "category",
      data: result.rowLabels,
      axisLabel: { color: "#9090b0", fontSize: 8 },
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
      show: result.colLabels.length > 1,
    },
    series: [
      {
        type: "heatmap",
        data: flat.map(([ci, ri, v]) => [ci, ri, v]),
        label: {
          show: result.rowLabels.length * result.colLabels.length <= 48,
          color: "#c0c0d8",
          fontSize: 8,
          formatter: (p: { value: [number, number, number] }) => {
            const v = p.value[2];
            return Number.isInteger(v) ? String(v) : v.toFixed(1);
          },
        },
      },
    ],
    tooltip: {
      formatter: (p: { value: [number, number, number] }) => {
        const [ci, ri, v] = p.value;
        return `${result.rowLabels[ri]} · ${result.colLabels[ci]}<br/>${v.toFixed(2)}`;
      },
    },
  };
}
