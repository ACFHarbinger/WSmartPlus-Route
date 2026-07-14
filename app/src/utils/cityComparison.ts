/**
 * City / graph comparison charts (§G.1.6).
 */
import { symlog } from "./symlog";
import { cityScaleLabel, parseLogPath } from "./simMetadata";
import type { DayLogEntry } from "../types";

export interface CityRunSlice {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
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

export interface CityComparisonSeries {
  labels: string[];
  profit: number[];
  overflowsSymlog: number[];
  kgkm: number[];
}

export function buildCityComparisonSeries(
  cityGroups: Array<[string, CityRunSlice[]]>
): CityComparisonSeries {
  const labels = cityGroups.map(([city]) => city);
  const profit = cityGroups.map(([, runs]) => {
    const vals = runs.flatMap((r) =>
      r.entries.map((e) => e.data.profit).filter((v): v is number => v != null)
    );
    return Math.max(mean(vals), 0.001);
  });
  const overflowsSymlog = cityGroups.map(([, runs]) => {
    const vals = runs.flatMap((r) =>
      r.entries.map((e) => e.data.overflows).filter((v): v is number => v != null)
    );
    return symlog(mean(vals));
  });
  const kgkm = cityGroups.map(([, runs]) => {
    const vals = runs.flatMap((r) =>
      r.entries.map((e) => e.data["kg/km"]).filter((v): v is number => v != null)
    );
    return Math.max(mean(vals), 0.001);
  });
  return { labels, profit, overflowsSymlog, kgkm };
}

/** ECharts option — log-scale bars preserving extreme values (§G.1.6). */
export function cityComparisonChartOption(series: CityComparisonSeries) {
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
      type: "log" as const,
      logBase: 10,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        name: "Mean profit (€)",
        type: "bar" as const,
        data: series.profit,
        itemStyle: { color: "#6366f1" },
      },
      {
        name: "Mean overflows (symlog)",
        type: "bar" as const,
        data: series.overflowsSymlog,
        itemStyle: { color: "#f87171" },
      },
      {
        name: "Mean kg/km",
        type: "bar" as const,
        data: series.kgkm,
        itemStyle: { color: "#34d399" },
      },
    ],
    tooltip: { trigger: "axis" as const },
  };
}
