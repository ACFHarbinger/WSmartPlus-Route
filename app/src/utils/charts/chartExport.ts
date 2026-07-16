import type { RefObject } from "react";
import type ReactECharts from "echarts-for-react";
import { toast } from "sonner";

function exportToast(success: boolean, filename: string, failReason = "Chart is not ready"): void {
  if (success) {
    toast.success("Chart exported", { description: filename });
    return;
  }
  toast.error("Export failed", { description: failReason });
}

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

/** PNG export with Sonner toast feedback (§G.7). */
export function exportChartPngWithToast(
  chartRef: RefObject<ReactECharts | null>,
  filename = "chart.png"
): void {
  exportToast(exportChartPng(chartRef, filename), filename);
}

/** SVG export with Sonner toast feedback (§G.7). */
export function exportChartSvgWithToast(
  chartRef: RefObject<ReactECharts | null>,
  filename = "chart.svg"
): void {
  exportToast(exportChartSvg(chartRef, filename), filename);
}

/** WebGL/canvas PNG export with Sonner toast feedback (§G.7). */
export function exportCanvasPngWithToast(
  canvas: HTMLCanvasElement | null | undefined,
  filename = "map.png"
): void {
  exportToast(exportCanvasPng(canvas, filename), filename, "Canvas is not ready");
}

/** Container canvas PNG export with Sonner toast feedback (§G.7). */
export function exportContainerCanvasPngWithToast(
  container: HTMLElement | null | undefined,
  filename = "canvas.png"
): void {
  exportToast(exportContainerCanvasPng(container, filename), filename, "Canvas is not ready");
}
