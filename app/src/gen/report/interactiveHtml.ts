/**
 * Interactive HTML exports (§H.5) — native port of `gen_interactive_html`
 * (simulation) and `gen_dataset_interactive_html` (dataset) from the archived
 * scripts, replacing the CDN-Plotly HTML files + injected `pareto_buttons.js`
 * with **self-contained** single files: ECharts is inlined into each page, so
 * they open offline; view toggles are plain HTML buttons driving `setOption`.
 *
 * Every option here is pure JSON (per-point hover text rides in `data[].name`
 * with a `{b}` tooltip template) so it can be serialised into the page.
 */
import echartsSrc from "echarts/dist/echarts.min.js?raw";
import { META, DATASET_CFG, disp, type GenTheme } from "../config";
import { rdYlGnColors } from "../charts/common";
import {
  mean,
  metricValues,
  paretoIndices,
  scenSub,
  scenarioLabel,
  variantColor,
  type AnalysisCtx,
  type SimRow,
} from "../data/simulation";
import type { NpzStatRow } from "../data/dataset";

const ECHARTS_SYMBOLS = ["circle", "rect", "diamond", "triangle", "arrow", "pin", "roundRect", "path://M0,0L10,10M10,0L0,10"];

export interface InteractiveView {
  label: string;
  /** setOption patch applied with replaceMerge on `series`. */
  patch: object;
}

export interface InteractivePage {
  title: string;
  height: number;
  option: object;
  views?: InteractiveView[];
}

/** Render a self-contained interactive chart page (inlined ECharts, no CDN). */
export function buildInteractiveHtmlPage(page: InteractivePage, theme: GenTheme): string {
  const optionJson = JSON.stringify(page.option);
  const viewsJson = JSON.stringify(page.views ?? []);
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>${esc(page.title)}</title>
<style>
  body { margin: 0; background: ${theme.bg}; color: ${theme.fg}; font-family: Helvetica, Arial, sans-serif; }
  h1 { font-size: 17px; text-align: center; margin: 14px 0 4px; }
  #chart { width: 100%; height: ${page.height}px; }
  .views { text-align: center; margin: 6px 0 12px; }
  .views button {
    background: ${theme.buttonBg}; border: 1px solid ${theme.buttonBorder}; color: ${theme.fg};
    font-size: 13px; padding: 5px 14px; margin: 0 4px; border-radius: 6px; cursor: pointer;
  }
  .views button.active { background: #3355cc; color: #ffffff; }
  .views button:hover { background: #5577ee; color: #ffffff; }
</style>
<script>${echartsSrc}</script>
</head>
<body>
<h1>${esc(page.title)}</h1>
<div class="views" id="views"></div>
<div id="chart"></div>
<script>
  var option = ${optionJson};
  var views = ${viewsJson};
  var chart = echarts.init(document.getElementById("chart"));
  chart.setOption(option);
  var host = document.getElementById("views");
  views.forEach(function (view, i) {
    var btn = document.createElement("button");
    btn.textContent = view.label;
    if (i === 0) btn.className = "active";
    btn.addEventListener("click", function () {
      host.querySelectorAll("button").forEach(function (b) { b.className = ""; });
      btn.className = "active";
      chart.setOption(view.patch, { replaceMerge: ["series", "visualMap"] });
    });
    host.appendChild(btn);
  });
  window.addEventListener("resize", function () { chart.resize(); });
</script>
</body>
</html>
`;
}

function baseAxes(theme: GenTheme, xName: string, yName: string) {
  const axis = (name: string) => ({
    type: "value",
    name,
    nameLocation: "middle",
    nameGap: 32,
    scale: true,
    nameTextStyle: { color: theme.axisLabelColor, fontSize: 13 },
    axisLabel: { color: theme.axisLabelColor },
    axisLine: { lineStyle: { color: theme.axisLabelColor } },
    splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
  });
  return { xAxis: axis(xName), yAxis: axis(yName) };
}

// ── Simulation: pareto_scatter_interactive ───────────────────────────────────

export function buildParetoInteractive(df: SimRow[], ctx: AnalysisCtx, theme: GenTheme): InteractivePage {
  const { scenarios, improvers, regions } = ctx;
  const regionSyms = new Map(regions.map(([c, n], i) => [`${c}|${n}`, ECHARTS_SYMBOLS[i % ECHARTS_SYMBOLS.length]]));
  const variants = [...new Set(df.map((r) => r.variant))].sort();
  const vcolors = new Map(variants.map((v, i) => [v, variantColor(v, i)]));
  const frontColors = new Map(
    scenarios.map((s, i) => [`${s.city}|${s.N}|${s.dist}`, META.scenario_colors[i % META.scenario_colors.length]])
  );

  const pointItem = (r: SimRow, s: (typeof scenarios)[number], starred = false) => ({
    value: [r.overflows, r.kgkm],
    name:
      `Constructor: ${disp(r.constructor)}<br>Selection: ${r.variant}<br>` +
      `Scenario: ${scenarioLabel(s)}<br>Improver: ${r.improver}<br>` +
      `Overflows: ${r.overflows.toFixed(1)}<br>kg/km: ${r.kgkm.toFixed(3)}` +
      (starred ? "<br><b>★ Pareto optimal</b>" : `<br>km: ${r.km.toFixed(0)}<br>Profit: ${r.profit.toFixed(0)}`),
    symbol: regionSyms.get(`${r.city}|${r.N}`),
    symbolSize: starred ? 14 : 9,
    itemStyle:
      r.improver === improvers[0]
        ? { color: vcolors.get(r.variant), borderColor: vcolors.get(r.variant), borderWidth: 1.5, opacity: 0.85 }
        : { color: "transparent", borderColor: vcolors.get(r.variant), borderWidth: 1.5, opacity: 0.85 },
  });

  const scatterSeries: object[] = [];
  const frontLineSeries: object[] = [];
  const frontMarkerSeries: object[] = [];
  for (const s of scenarios) {
    const sub = scenSub(df, s);
    if (!sub.length) continue;
    const byVariant = new Map<string, SimRow[]>();
    for (const r of sub) (byVariant.get(r.variant) ?? byVariant.set(r.variant, []).get(r.variant)!).push(r);
    for (const [variant, rows] of byVariant) {
      scatterSeries.push({
        type: "scatter",
        name: `${variant} (${s.dist.slice(0, 3)})`,
        data: rows.map((r) => pointItem(r, s)),
      });
    }
    const front = paretoIndices(metricValues(sub, "overflows"), metricValues(sub, "kgkm"));
    const pts = [...front].map((i) => [sub[i].overflows, sub[i].kgkm] as [number, number]).sort((a, b) => a[0] - b[0]);
    if (pts.length) {
      const step: [number, number][] = [pts[0]];
      for (let j = 1; j < pts.length; j++) step.push([pts[j][0], pts[j - 1][1]], [pts[j][0], pts[j][1]]);
      frontLineSeries.push({
        type: "line",
        name: `Pareto — ${scenarioLabel(s)}`,
        data: step,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 2, type: "dashed", color: frontColors.get(`${s.city}|${s.N}|${s.dist}`) },
      });
    }
    frontMarkerSeries.push({
      type: "scatter",
      name: `★ Pareto — ${scenarioLabel(s)}`,
      data: [...front].map((i) => pointItem(sub[i], s, true)),
    });
  }

  const { xAxis, yAxis } = baseAxes(theme, `Overflows (${ctx.nDays} days)`, "kg/km");
  const legend = { type: "scroll", bottom: 4, textStyle: { color: theme.fg, fontSize: 11 } };
  const tooltip = { trigger: "item", formatter: "{b}" };
  return {
    title: `Overflow vs Efficiency — All Runs (${ctx.nDays} days; hover for details)`,
    height: 750,
    option: {
      backgroundColor: theme.bg,
      tooltip,
      legend,
      grid: { left: 70, right: 30, top: 30, bottom: 90 },
      xAxis,
      yAxis,
      series: [...scatterSeries, ...frontLineSeries],
    },
    views: [
      { label: "All Points", patch: { series: [...scatterSeries, ...frontLineSeries] } },
      { label: "Pareto Front Only", patch: { series: [...frontLineSeries, ...frontMarkerSeries] } },
    ],
  };
}

// ── Simulation: strategy_bubble_interactive ──────────────────────────────────

export function buildStrategyBubbleInteractive(dfm: SimRow[], ctx: AnalysisCtx, theme: GenTheme): InteractivePage {
  const { scenarios, improvers, regions } = ctx;
  const regionSyms = new Map(regions.map(([c, n], i) => [`${c}|${n}`, ECHARTS_SYMBOLS[i % ECHARTS_SYMBOLS.length]]));
  // mean over constructors per (scenario, strategy, improver) — ports the groupby
  const groups = new Map<string, SimRow[]>();
  for (const r of dfm) {
    const key = [r.city, r.N, r.dist, r.strategy, r.improver].join("|");
    (groups.get(key) ?? groups.set(key, []).get(key)!).push(r);
  }
  const series: object[] = [];
  for (const strat of ctx.strategies) {
    for (const s of scenarios) {
      const data: object[] = [];
      for (const imp of improvers) {
        const grp = groups.get([s.city, s.N, s.dist, strat, imp].join("|"));
        if (!grp?.length) continue;
        const ov = mean(grp.map((r) => r.overflows));
        const eff = mean(grp.map((r) => r.kgkm));
        data.push({
          value: [ov, eff],
          name:
            `Strategy: ${strat}<br>Scenario: ${scenarioLabel(s)}<br>` +
            `Improver: ${imp}<br>Mean overflows: ${ov.toFixed(2)}<br>Mean kg/km: ${eff.toFixed(3)}`,
          symbol: regionSyms.get(`${s.city}|${s.N}`),
          symbolSize: s.N / 15 + 8,
          itemStyle:
            imp === improvers[0]
              ? { color: META.strategy_colors[strat] ?? "gray", opacity: 0.85, borderColor: theme.fg, borderWidth: 1 }
              : {
                  color: "transparent",
                  borderColor: META.strategy_colors[strat] ?? "gray",
                  borderWidth: 1.6,
                  opacity: 0.9,
                },
        });
      }
      if (data.length) {
        series.push({ type: "scatter", name: `${strat} (${s.dist.slice(0, 3)})`, data });
      }
    }
  }
  const { xAxis, yAxis } = baseAxes(theme, "Mean overflows", "Mean kg/km");
  return {
    title: "Strategy Trade-off Bubble Chart (bubble size ∝ N; open marker = second improver)",
    height: 650,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { type: "scroll", bottom: 4, textStyle: { color: theme.fg, fontSize: 11 } },
      grid: { left: 70, right: 30, top: 30, bottom: 80 },
      xAxis,
      yAxis,
      series,
    },
  };
}

// ── Simulation: policy_heatmap_interactive ───────────────────────────────────

export function buildPolicyHeatmapInteractive(df: SimRow[], ctx: AnalysisCtx, theme: GenTheme): InteractivePage {
  const combos: [string, string, string][] = [];
  const seen = new Set<string>();
  for (const r of df) {
    const key = `${r.variant}|${r.constructor}|${r.improver}`;
    if (!seen.has(key)) {
      seen.add(key);
      combos.push([r.variant, r.constructor, r.improver]);
    }
  }
  combos.sort((a, b) => a[0].localeCompare(b[0]) || a[1].localeCompare(b[1]) || a[2].localeCompare(b[2]));
  const rowLabels = combos.map(([v, c, i]) => `${v} · ${disp(c)} · ${i}`);
  const colLabels = ctx.scenarios.map(scenarioLabel);

  const matFor = (metric: "overflows" | "kgkm") => {
    const data: [number, number, number][] = [];
    combos.forEach(([v, c, i], ri) => {
      const subP = df.filter((r) => r.variant === v && r.constructor === c && r.improver === i);
      ctx.scenarios.forEach((s, ci) => {
        const vals = metricValues(scenSub(subP, s), metric);
        if (vals.length) data.push([ci, ri, Number(mean(vals).toFixed(3))]);
      });
    });
    return data;
  };
  const ovData = matFor("overflows");
  const kgData = matFor("kgkm");
  const bounds = (data: [number, number, number][]) => {
    const vs = data.map((d) => d[2]);
    return { min: Math.min(...vs), max: Math.max(...vs) };
  };
  const visualMapFor = (data: [number, number, number][], reversed: boolean) => ({
    ...bounds(data),
    calculable: true,
    orient: "vertical",
    right: 8,
    top: "center",
    inRange: { color: rdYlGnColors(reversed) },
    textStyle: { color: theme.fg },
  });
  const seriesFor = (data: [number, number, number][]) => [
    { type: "heatmap", data, emphasis: { disabled: true } },
  ];

  const height = Math.max(700, 22 * combos.length);
  const catAxis = (data: string[], isX: boolean) => ({
    type: "category",
    data,
    ...(isX ? {} : { inverse: true }),
    axisLabel: { color: theme.axisLabelColor, fontSize: isX ? 11 : 10, interval: 0, ...(isX ? { rotate: 30 } : {}) },
  });
  return {
    title: "Policy × Scenario Heatmap",
    height,
    option: {
      backgroundColor: theme.bg,
      tooltip: {
        position: "top",
        formatter: "Policy: {b}<br>Value: {c}",
      },
      grid: { left: 250, right: 100, top: 20, bottom: 90 },
      xAxis: catAxis(colLabels, true),
      yAxis: catAxis(rowLabels, false),
      visualMap: visualMapFor(ovData, true),
      series: seriesFor(ovData),
    },
    views: [
      { label: "Overflows", patch: { visualMap: visualMapFor(ovData, true), series: seriesFor(ovData) } },
      { label: "kg/km", patch: { visualMap: visualMapFor(kgData, false), series: seriesFor(kgData) } },
    ],
  };
}

// ── Dataset: npz_stats / waste_distribution / city_network_comparison ────────

const CITY_LABELS = DATASET_CFG.city_labels;
const DIST_LABELS = DATASET_CFG.dist_labels;
const CITY_COLORS = DATASET_CFG.city_colors;
const DIST_COLORS = DATASET_CFG.dist_colors;

export function buildNpzStatsInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const series: object[] = [];
  for (const city of cities) {
    for (const dist of dists) {
      const rows = sub30.filter((r) => r.city === city && r.dist === dist);
      if (!rows.length) continue;
      series.push({
        type: "scatter",
        name: `${CITY_LABELS[city] ?? city} — ${DIST_LABELS[dist] ?? dist}`,
        data: rows.map((r) => ({
          value: [r.mean_kg, r.std_kg],
          name:
            `${CITY_LABELS[city] ?? city}<br>N=${r.N}<br>Dist: ${DIST_LABELS[dist] ?? dist}<br>` +
            `Mean kg: ${r.mean_kg.toFixed(3)}<br>Std kg: ${r.std_kg.toFixed(3)}<br>` +
            `Max kg: ${r.max_kg.toFixed(1)}<br>Skewness: ${r.skewness.toFixed(3)}`,
          symbol: (DIST_LABELS[dist] ?? dist) === "Empirical" ? "circle" : "rect",
          symbolSize: r.N / 20 + 10,
          itemStyle: { color: CITY_COLORS[city] ?? "#a0a0a0", opacity: 0.85, borderColor: theme.fg, borderWidth: 1 },
        })),
      });
    }
  }
  const { xAxis, yAxis } = baseAxes(theme, "Mean waste (kg/bin/day)", "Std waste (kg/bin/day)");
  return {
    title: "NPZ Dataset Statistics — Mean vs Std Waste (30-day horizon)",
    height: 600,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { bottom: 4, textStyle: { color: theme.fg, fontSize: 11 } },
      grid: { left: 80, right: 30, top: 30, bottom: 70 },
      xAxis,
      yAxis,
      series,
    },
  };
}

export function buildWasteDistributionInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const series: object[] = [];
  cities.forEach((city, ci) => {
    const left = 7 + ci * (90 / cities.length);
    grids.push({ left: `${left}%`, top: "14%", width: `${90 / cities.length - 6}%`, height: "70%" });
    titles.push({
      text: CITY_LABELS[city] ?? city,
      left: `${left + (90 / cities.length - 6) / 2}%`,
      top: "6%",
      textAlign: "center",
      textStyle: { color: theme.fg, fontSize: 13 },
    });
    const rows = sub30.filter((r) => r.city === city).sort((a, b) => a.N - b.N);
    const ns = [...new Set(rows.map((r) => r.N))];
    xAxes.push({
      type: "category",
      gridIndex: ci,
      data: ns.map((n) => `N=${n}`),
      axisLabel: { color: theme.axisLabelColor, fontSize: 10 },
    });
    yAxes.push({
      type: "value",
      gridIndex: ci,
      name: ci === 0 ? "Mean waste (kg/bin/day)" : "",
      nameTextStyle: { color: theme.axisLabelColor },
      axisLabel: { color: theme.axisLabelColor },
      splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
    });
    for (const dist of dists) {
      const byN = new Map(rows.filter((r) => r.dist === dist).map((r) => [r.N, r]));
      series.push({
        type: "bar",
        name: DIST_LABELS[dist] ?? dist,
        xAxisIndex: ci,
        yAxisIndex: ci,
        data: ns.map((n) => {
          const r = byN.get(n);
          return r
            ? {
                value: r.mean_kg,
                name:
                  `${CITY_LABELS[city] ?? city}, N=${n}, ${DIST_LABELS[dist] ?? dist}<br>` +
                  `Mean: ${r.mean_kg.toFixed(3)} kg<br>Std: ${r.std_kg.toFixed(3)} kg`,
              }
            : null;
        }),
        itemStyle: { color: DIST_COLORS[dist] ?? "gray", opacity: 0.85 },
      });
    }
  });
  return {
    title: "Waste Distribution Statistics per City and Network Size",
    height: 550,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { bottom: 2, textStyle: { color: theme.fg, fontSize: 11 } },
      title: titles,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    },
  };
}

export function buildCityNetworkComparisonInteractive(npz: NpzStatRow[], theme: GenTheme): InteractivePage {
  const sub30 = npz.filter((r) => r.horizon === 30);
  const cities = [...new Set(sub30.map((r) => r.city))].sort();
  const dists = [...new Set(sub30.map((r) => r.dist))].sort();
  const metrics: [keyof NpzStatRow, string][] = [
    ["mean_kg", "Mean Waste (kg/bin)"],
    ["std_kg", "Std Waste (kg/bin)"],
    ["skewness", "Skewness"],
    ["max_kg", "Max Waste (kg)"],
  ];
  const grids: object[] = [];
  const xAxes: object[] = [];
  const yAxes: object[] = [];
  const titles: object[] = [];
  const series: object[] = [];
  const categories = sub30
    .sort((a, b) => a.city.localeCompare(b.city) || a.N - b.N)
    .map((r) => `${(CITY_LABELS[r.city] ?? r.city).slice(0, 3)} N=${r.N}`)
    .filter((v, i, arr) => arr.indexOf(v) === i);

  metrics.forEach(([metric, label], mi) => {
    const col = mi % 2;
    const row = Math.floor(mi / 2);
    grids.push({ left: `${7 + col * 47}%`, top: `${11 + row * 45}%`, width: "40%", height: "33%" });
    titles.push({
      text: label,
      left: `${7 + col * 47 + 20}%`,
      top: `${5 + row * 45}%`,
      textAlign: "center",
      textStyle: { color: theme.fg, fontSize: 12 },
    });
    xAxes.push({
      type: "category",
      gridIndex: mi,
      data: categories,
      axisLabel: { color: theme.axisLabelColor, fontSize: 9, rotate: 30, interval: 0 },
    });
    yAxes.push({
      type: "value",
      gridIndex: mi,
      scale: true,
      axisLabel: { color: theme.axisLabelColor, fontSize: 10 },
      splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
    });
    for (const city of cities) {
      for (const dist of dists) {
        const rows = sub30.filter((r) => r.city === city && r.dist === dist);
        if (!rows.length) continue;
        const byCat = new Map(
          rows.map((r) => [`${(CITY_LABELS[city] ?? city).slice(0, 3)} N=${r.N}`, r])
        );
        series.push({
          type: "bar",
          name: `${CITY_LABELS[city] ?? city} ${DIST_LABELS[dist] ?? dist}`,
          xAxisIndex: mi,
          yAxisIndex: mi,
          data: categories.map((cat) => {
            const r = byCat.get(cat);
            return r
              ? {
                  value: r[metric] as number,
                  name:
                    `${CITY_LABELS[city] ?? city}<br>N=${r.N}<br>Dist: ${DIST_LABELS[dist] ?? dist}<br>` +
                    `${String(metric)}: ${(r[metric] as number).toFixed(3)}`,
                }
              : null;
          }),
          itemStyle: { color: CITY_COLORS[city] ?? "gray", opacity: dist === "emp" ? 0.85 : 0.55 },
        });
      }
    }
  });
  return {
    title: "City & Network Comparison — NPZ Dataset Statistics",
    height: 800,
    option: {
      backgroundColor: theme.bg,
      tooltip: { trigger: "item", formatter: "{b}" },
      legend: { type: "scroll", bottom: 2, textStyle: { color: theme.fg, fontSize: 10 } },
      title: titles,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    },
  };
}
