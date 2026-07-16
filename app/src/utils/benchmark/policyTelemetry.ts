/**
 * Policy telemetry parsing and ECharts builders (§A.3).
 *
 * Mirrors ``logic/src/ui/components/policy_viz.py`` dispatch logic for the Studio.
 */
import type { EChartsOption } from "echarts";
import type { PolicyVizEntry, PolicyVizType } from "../../types";

export const POLICY_VIZ_MARKER = "POLICY_VIZ_START:";

const CHART_COLORS = ["#60a5fa", "#fbbf24", "#4ade80", "#f87171", "#c084fc", "#94a3b8"];

function splitFirstCommas(content: string, count: number): string[] | null {
  let rest = content;
  const parts: string[] = [];
  for (let i = 0; i < count; i++) {
    const idx = rest.indexOf(",");
    if (idx === -1) return null;
    parts.push(rest.slice(0, idx));
    rest = rest.slice(idx + 1);
  }
  parts.push(rest);
  return parts;
}

export function parsePolicyVizLine(line: string): PolicyVizEntry | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(POLICY_VIZ_MARKER)) return null;
  const content = trimmed.slice(POLICY_VIZ_MARKER.length);
  const parts = splitFirstCommas(content, 4);
  if (!parts || parts.length < 5) return null;
  const policy = parts[0]!.trim();
  const sample_id = Number.parseInt(parts[1]!.trim(), 10);
  const day = Number.parseInt(parts[2]!.trim(), 10);
  const policy_type = (parts[3]!.trim() || "generic") as PolicyVizType;
  try {
    const data = JSON.parse(parts[4]!.trim()) as PolicyVizEntry["data"];
    if (!Number.isFinite(sample_id) || !Number.isFinite(day)) return null;
    return { policy, sample_id, day, policy_type, data };
  } catch {
    return null;
  }
}

export function ema(values: number[], window = 5): number[] {
  if (window <= 1 || values.length === 0) return values;
  const alpha = 2 / (window + 1);
  const out = [values[0]!];
  for (let i = 1; i < values.length; i++) {
    out.push(alpha * values[i]! + (1 - alpha) * out[i - 1]!);
  }
  return out;
}

function asNumbers(values: Array<number | string | boolean> | undefined): number[] {
  if (!values?.length) return [];
  return values.map((v) => (typeof v === "number" ? v : Number(v))).filter((n) => Number.isFinite(n));
}

function barCounts(values: Array<number | string | boolean> | undefined): { labels: string[]; counts: number[] } {
  const counts = new Map<string, number>();
  for (const v of values ?? []) {
    const key = String(v);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return {
    labels: [...counts.keys()],
    counts: [...counts.values()],
  };
}

function baseLineOption(
  title: string,
  x: number[],
  series: Array<{ name: string; data: number[]; dashed?: boolean | undefined }>,
  theme: "dark" | "light",
  logScale = false
): EChartsOption {
  return {
    backgroundColor: "transparent",
    title: { text: title, left: 8, top: 4, textStyle: { color: theme === "dark" ? "#d1d5db" : "#374151", fontSize: 12 } },
    grid: { left: 48, right: 16, top: 36, bottom: 28 },
    tooltip: { trigger: "axis" },
    legend: { top: 4, right: 8, textStyle: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 } },
    xAxis: {
      type: "category",
      data: x.map(String),
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: theme === "dark" ? "#1f2937" : "#e5e7eb" } },
    },
    series: series.map((s, i) => ({
      name: s.name,
      type: "line",
      smooth: true,
      showSymbol: false,
      data: s.data,
      lineStyle: { color: CHART_COLORS[i % CHART_COLORS.length], width: 2, type: s.dashed ? "dotted" : "solid" },
    })),
  };
}

function baseBarOption(
  title: string,
  labels: string[],
  counts: number[],
  theme: "dark" | "light",
  color = CHART_COLORS[0]
): EChartsOption {
  return {
    backgroundColor: "transparent",
    title: { text: title, left: 8, top: 4, textStyle: { color: theme === "dark" ? "#d1d5db" : "#374151", fontSize: 12 } },
    grid: { left: 48, right: 16, top: 36, bottom: 40 },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 9, rotate: 20 },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: theme === "dark" ? "#9ca3af" : "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: theme === "dark" ? "#1f2937" : "#e5e7eb" } },
    },
    series: [{ type: "bar", data: counts, itemStyle: { color } }],
  };
}

export function policyVizChartOptions(
  entry: PolicyVizEntry,
  theme: "dark" | "light",
  logScale = false,
  smoothWindow = 5
): EChartsOption[] {
  const { data, policy_type } = entry;
  const iterations = asNumbers(data.iteration);
  const x = iterations.length ? iterations : asNumbers(data.generation).length
    ? asNumbers(data.generation)
    : asNumbers(data.restart).length
      ? asNumbers(data.restart)
      : [...Array(Math.max(...Object.values(data).map((v) => v.length), 0)).keys()];

  switch (policy_type) {
    case "alns": {
      const charts: EChartsOption[] = [];
      const best = ema(asNumbers(data.best_cost), smoothWindow);
      const current = ema(asNumbers(data.current_cost), smoothWindow);
      const series: Array<{ name: string; data: number[]; dashed?: boolean }> = [
        { name: "Best Cost", data: best },
      ];
      if (current.length) series.push({ name: "Current Cost", data: current, dashed: true });
      charts.push(baseLineOption("ALNS Cost Trajectory", x, series, theme, logScale));
      const temps = asNumbers(data.temperature);
      if (temps.length) {
        charts.push(baseLineOption("SA Temperature", x.slice(0, temps.length), [{ name: "T", data: temps }], theme, false));
      }
      const dCounts = barCounts(data.d_idx);
      const rCounts = barCounts(data.r_idx);
      if (dCounts.labels.length) {
        charts.push(baseBarOption("Destroy Operator Usage", dCounts.labels, dCounts.counts, theme, CHART_COLORS[2]));
      }
      if (rCounts.labels.length) {
        charts.push(baseBarOption("Repair Operator Usage", rCounts.labels, rCounts.counts, theme, CHART_COLORS[4]));
      }
      return charts;
    }
    case "hgs": {
      const gens = asNumbers(data.generation);
      const gx = gens.length ? gens : x;
      const series = [
        { name: "Best", data: ema(asNumbers(data.best_cost), smoothWindow) },
        { name: "Mean", data: ema(asNumbers(data.mean_cost), smoothWindow) },
        { name: "Worst", data: ema(asNumbers(data.worst_cost), smoothWindow) },
      ].filter((s) => s.data.length > 0);
      return [baseLineOption("HGS Population Fitness", gx, series, theme, logScale)];
    }
    case "aco": {
      const series = [
        { name: "Global Best", data: ema(asNumbers(data.global_best_cost), smoothWindow) },
        { name: "Iter Best", data: ema(asNumbers(data.iter_best_cost), smoothWindow) },
        { name: "τ mean", data: asNumbers(data.tau_mean) },
        { name: "τ max", data: asNumbers(data.tau_max) },
      ].filter((s) => s.data.length > 0);
      return [baseLineOption("ACO Convergence & Pheromone", x, series, theme, logScale)];
    }
    case "ils": {
      const restarts = asNumbers(data.restart);
      const rx = restarts.length ? restarts : x;
      const series = [
        { name: "Best Cost", data: ema(asNumbers(data.best_cost), smoothWindow) },
        { name: "Candidate", data: ema(asNumbers(data.candidate_cost), smoothWindow), dashed: true },
      ].filter((s) => s.data.length > 0);
      const charts = [baseLineOption("ILS Cost per Restart", rx, series, theme, logScale)];
      const perturb = barCounts(data.perturb_mode);
      if (perturb.labels.length) {
        charts.push(baseBarOption("Perturbation Usage", perturb.labels, perturb.counts, theme, CHART_COLORS[1]));
      }
      return charts;
    }
    case "selector": {
      const calls = [...Array(asNumbers(data.n_selected).length).keys()];
      const charts: EChartsOption[] = [
        baseBarOption("Bins Selected per Call", calls.map(String), asNumbers(data.n_selected), theme, CHART_COLORS[0]),
      ];
      const meanFill = asNumbers(data.mean_fill);
      if (meanFill.length) {
        charts.push(baseLineOption("Mean Fill at Selection", calls, [{ name: "Fill", data: meanFill }], theme, false));
      }
      return charts;
    }
    default: {
      const numericKeys = Object.keys(data).filter(
        (k) => k !== "iteration" && k !== "op_name" && data[k]?.[0] !== undefined && typeof data[k]![0] === "number"
      );
      const opCounts = barCounts(data.op_name);
      const charts: EChartsOption[] = [];
      if (opCounts.labels.length) {
        charts.push(baseBarOption("Operator Usage", opCounts.labels, opCounts.counts, theme, CHART_COLORS[0]));
      }
      if (numericKeys.length) {
        charts.push(
          baseLineOption(
            "Policy Metrics",
            x,
            numericKeys.slice(0, 5).map((key, i) => ({
              name: key,
              data: ema(asNumbers(data[key]), smoothWindow),
              dashed: i > 0,
            })),
            theme,
            logScale
          )
        );
      }
      return charts.length ? charts : [];
    }
  }
}

/** Longest metric series length — used to pick the newest streaming snapshot. */
export function policyVizDataLen(data: PolicyVizEntry["data"]): number {
  if (!data || typeof data !== "object") return 0;
  return Math.max(0, ...Object.values(data).map((v) => (Array.isArray(v) ? v.length : 0)));
}

/** Parse all ``POLICY_VIZ_START:`` markers from process stdout (§A.3 Option B). */
export function collectPolicyVizFromLogLines(lines: string[]): PolicyVizEntry[] {
  const entries: PolicyVizEntry[] = [];
  for (const line of lines) {
    const parsed = parsePolicyVizLine(line);
    if (parsed) entries.push(parsed);
  }
  return entries;
}

/** Unique policy names from parsed telemetry entries, stable order. */
export function uniquePolicyVizPolicies(entries: PolicyVizEntry[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const e of entries) {
    if (!seen.has(e.policy)) {
      seen.add(e.policy);
      out.push(e.policy);
    }
  }
  return out;
}

export function filterPolicyVizEntries(
  entries: PolicyVizEntry[],
  policy: string | null,
  sampleId: number | null,
  day: number | null
): PolicyVizEntry[] {
  return entries.filter(
    (e) =>
      (policy === null || e.policy === policy) &&
      (sampleId === null || e.sample_id === sampleId) &&
      (day === null || e.day === day)
  );
}

export function policyVizTypeLabel(type: PolicyVizType): string {
  const labels: Record<PolicyVizType, string> = {
    alns: "ALNS",
    hgs: "HGS",
    aco: "ACO",
    ils: "ILS",
    selector: "Selection",
    generic: "Generic",
  };
  return labels[type] ?? type;
}
