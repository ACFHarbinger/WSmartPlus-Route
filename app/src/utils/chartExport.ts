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

/** Export an ECharts instance as SVG (§G.7). */
export function exportChartSvg(
  chartRef: RefObject<ReactECharts | null>,
  filename = "chart.svg"
): boolean {
  const instance = chartRef.current?.getEchartsInstance();
  if (!instance) return false;

  const svg = instance.renderToSVGString();
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.endsWith(".svg") ? filename : `${filename}.svg`;
  link.click();
  URL.revokeObjectURL(url);
  return true;
}

/** Export a WebGL/canvas element as PNG (deck.gl tile maps). */
export function exportCanvasPng(
  canvas: HTMLCanvasElement | null | undefined,
  filename = "map.png"
): boolean {
  if (!canvas) return false;
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = filename;
  link.click();
  return true;
}

/** Export the first canvas inside a container (Sigma.js, R3F, deck.gl). */
export function exportContainerCanvasPng(
  container: HTMLElement | null | undefined,
  filename = "canvas.png"
): boolean {
  const canvas = container?.querySelector("canvas");
  return exportCanvasPng(canvas, filename);
}
