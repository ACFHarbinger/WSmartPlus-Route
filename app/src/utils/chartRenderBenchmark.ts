/**
 * Measure ECharts first-paint render latency (§G.7 performance budget).
 */

import * as echarts from "echarts";

const CHART_RENDER_BUDGET_MS = 500;

export function getChartRenderBudgetMs(): number {
  return CHART_RENDER_BUDGET_MS;
}

/** Render a representative 50×8 bar chart off-screen; returns elapsed ms. */
export async function runChartRenderBenchmark(): Promise<number> {
  const div = document.createElement("div");
  div.style.cssText = "position:fixed;left:-9999px;top:0;width:800px;height:400px;visibility:hidden";
  document.body.appendChild(div);

  const chart = echarts.init(div, undefined, { renderer: "canvas" });
  const categories = Array.from({ length: 50 }, (_, i) => `P${i + 1}`);
  const series = Array.from({ length: 8 }, (_, si) => ({
    name: `S${si + 1}`,
    type: "bar" as const,
    data: categories.map((_, ci) => 10 + si * 3 + (ci % 7)),
    itemStyle: { color: `hsl(${si * 40}, 70%, 60%)` },
  }));

  const t0 = performance.now();
  chart.setOption({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 30, bottom: 50 },
    legend: { textStyle: { color: "#9090b0" } },
    xAxis: { type: "category", data: categories },
    yAxis: { type: "value" },
    series,
  });

  await new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });

  const ms = Math.round(performance.now() - t0);
  chart.dispose();
  document.body.removeChild(div);
  return ms;
}
