/**
 * City / graph comparison charts (§G.1.6).
 */
import { errorBarBounds, groupedBarWhiskerX } from "../charts/chartLogScale";
import { symlog } from "../charts/symlog";
import { cityScaleLabel, parseLogPath } from "../sim/simMetadata";
import type { DayLogEntry } from "../../types";

export interface CityRunSlice {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function fmt(n: number, d = 2) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

/** Group loaded runs by city/scale label (RM-100, RM-170, FFZ-350, …). */
export function groupRunsByCity(runs: CityRunSlice[]): Array<[string, CityRunSlice[]]> {
  const map = new Map<string, CityRunSlice[]>();
  for (const run of runs) {
    const label = cityScaleLabel(parseLogPath(run.path));
    const list = map.get(label) ?? [];
    list.push(run);
    map.set(label, list);
  }
  return [...map.entries()];
}

/** Group DuckDB ``run_label`` values by city/scale (parses label as a log path). */
export function groupRunLabelsByCity(labels: string[]): Array<[string, string[]]> {
  const map = new Map<string, string[]>();
  for (const label of labels) {
    const city = cityScaleLabel(parseLogPath(label));
    const list = map.get(city) ?? [];
    list.push(label);
    map.set(city, list);
  }
  return [...map.entries()];
}

/** City/scale label for a portfolio ``run_label`` (parses label as a log path). */
export function cityScaleFromRunLabel(label: string): string {
  return cityScaleLabel(parseLogPath(label));
}

/** Expand global city brush to ``run_label`` list for SQL sync (§G.6). */
export function resolveBrushedRunLabels(
  runLabels: string[],
  runLabel: string | null,
  brushedCity: string | null
): string[] | null {
  if (runLabel) return [runLabel];
  if (!brushedCity || !runLabels.length) return null;
  const group = groupRunLabelsByCity(runLabels).find(([city]) => city === brushedCity);
  return group?.[1] ?? null;
}

export interface CityComparisonSeries {
  labels: string[];
  profit: number[];
  profitStd: number[];
  overflows: number[];
  overflowsStd: number[];
  kgkm: number[];
  kgkmStd: number[];
}

export function buildCityComparisonSeries(
  cityGroups: Array<[string, CityRunSlice[]]>
): CityComparisonSeries {
  const labels = cityGroups.map(([city]) => city);
  const profit: number[] = [];
  const profitStd: number[] = [];
  const overflows: number[] = [];
  const overflowsStd: number[] = [];
  const kgkm: number[] = [];
  const kgkmStd: number[] = [];

  for (const [, runs] of cityGroups) {
    const profitVals = runs.flatMap((r) =>
      r.entries.map((e) => e.data.profit).filter((v): v is number => v != null)
    );
    const overflowVals = runs.flatMap((r) =>
      r.entries.map((e) => e.data.overflows).filter((v): v is number => v != null)
    );
    const kgkmVals = runs.flatMap((r) =>
      r.entries.map((e) => e.data["kg/km"]).filter((v): v is number => v != null)
    );
    profit.push(mean(profitVals));
    profitStd.push(std(profitVals));
    overflows.push(mean(overflowVals));
    overflowsStd.push(std(overflowVals));
    kgkm.push(mean(kgkmVals));
    kgkmStd.push(std(kgkmVals));
  }

  return { labels, profit, profitStd, overflows, overflowsStd, kgkm, kgkmStd };
}

const CITY_METRIC_SERIES = [
  { key: "profit", stdKey: "profitStd", label: "Mean profit (€)", symlog: false },
  { key: "overflows", stdKey: "overflowsStd", label: "Mean overflows", symlog: true },
  { key: "kgkm", stdKey: "kgkmStd", label: "Mean kg/km", symlog: false },
] as const;

/** ECharts option — log or linear bars; symlog-overflows when log scale on (§G.1.6 / §G.7). */
export function cityComparisonChartOption(
  series: CityComparisonSeries,
  opts?: { logScale?: boolean; showErrorBars?: boolean }
) {
  const logScale = opts?.logScale ?? false;
  const showErrorBars = opts?.showErrorBars ?? false;
  const display = (v: number) => (logScale ? Math.max(v, 0.001) : v);

  const barSeries = CITY_METRIC_SERIES.map((m, seriesIdx) => {
    const values = series[m.key as keyof CityComparisonSeries] as number[];
    const data =
      logScale && m.symlog
        ? values.map((v) => symlog(v))
        : logScale
          ? values.map(display)
          : values;
    return {
      name: logScale && m.symlog ? `${m.label} (symlog)` : m.label,
      type: "bar" as const,
      data,
      itemStyle: {
        color: seriesIdx === 0 ? "#6366f1" : seriesIdx === 1 ? "#f87171" : "#34d399",
      },
    };
  });

  const errorBarPoints = CITY_METRIC_SERIES.flatMap((m, seriesIdx) =>
    series.labels.map((_, catIdx) => ({
      catIdx,
      seriesIdx,
      mean: (series[m.key as keyof CityComparisonSeries] as number[])[catIdx],
      std: (series[m.stdKey as keyof CityComparisonSeries] as number[])[catIdx],
      metricKey: m.key === "kgkm" ? "kg/km" : m.key,
      symlog: logScale && m.symlog,
    }))
  );

  const errorBarSeries = showErrorBars
    ? [
        {
          type: "custom" as const,
          renderItem: (
            params: { dataIndex: number },
            api: {
              coord: (v: [number, number]) => [number, number];
              size: (v: [number, number]) => [number, number];
              style: (s: object) => object;
            }
          ) => {
            const point = errorBarPoints[params.dataIndex];
            const bounds = errorBarBounds(
              point.mean,
              point.std,
              point.metricKey,
              logScale,
              point.symlog
            );
            const x = groupedBarWhiskerX(
              api,
              point.catIdx,
              point.seriesIdx,
              CITY_METRIC_SERIES.length,
              bounds.center
            );
            const yTop = api.coord([point.catIdx, bounds.high])[1];
            const yBot = api.coord([point.catIdx, bounds.low])[1];
            const cap = 4;
            return {
              type: "group",
              children: [
                {
                  type: "line",
                  shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                  style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                },
                {
                  type: "line",
                  shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                  style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                },
                {
                  type: "line",
                  shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                  style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                },
              ],
            };
          },
          data: errorBarPoints.map((_, i) => i),
          z: 10,
        },
      ]
    : [];

  return {
    backgroundColor: "transparent",
    legend: { textStyle: { color: "#9090b0", fontSize: 10 } },
    grid: { left: 50, right: 10, top: 30, bottom: 40 },
    xAxis: {
      type: "category" as const,
      data: series.labels,
      axisLabel: { color: "#9090b0", fontSize: 9 },
    },
    yAxis: {
      type: (logScale ? "log" : "value") as "log" | "value",
      logBase: 10,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    series: [...barSeries, ...errorBarSeries],
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: unknown[]) => {
        const items = params as Array<{ seriesName: string; dataIndex: number; value: number }>;
        const idx = items[0]?.dataIndex ?? 0;
        const city = series.labels[idx];
        const lines = CITY_METRIC_SERIES.map((m) => {
          const meanVal = (series[m.key as keyof CityComparisonSeries] as number[])[idx];
          const stdVal = (series[m.stdKey as keyof CityComparisonSeries] as number[])[idx];
          const name = logScale && m.symlog ? `${m.label} (symlog)` : m.label;
          return `${name}: ${fmt(meanVal)} ± ${fmt(stdVal)}`;
        });
        return `${city}<br/>${lines.join("<br/>")}`;
      },
    },
  };
}
