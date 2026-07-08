import type { RefObject } from "react";
import type ReactECharts from "echarts-for-react";

/** Export an ECharts instance as a PNG download (§G.7 chart export). */
export function exportChartPng(
  chartRef: RefObject<ReactECharts | null>,
  filename = "chart.png"
): boolean {
  const instance = chartRef.current?.getEchartsInstance();
  if (!instance) return false;

  const dataUrl = instance.getDataURL({
    type: "png",
    pixelRatio: 2,
    backgroundColor: "#1a1a2e",
  });

  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
  return true;
}
