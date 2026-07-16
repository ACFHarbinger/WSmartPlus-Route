/**
 * Shared chart machinery for the native report/deck generator (§H.2).
 *
 * Panel-grid layout (ports `_panel_grid`), theme-aware base styling, error
 * bar + step-front helpers, and headless ECharts → PNG rendering so figure
 * export produces exactly what the preview shows.
 */
import * as echarts from "echarts";
import type { EChartsOption, SeriesOption } from "echarts";
import type { GenTheme } from "../config";
import { symlog } from "../../utils/charts/symlog";

export interface ChartSpec {
  option: EChartsOption;
  width: number;
  height: number;
  /** Panel background override (radar uses its own dark palette). */
  background?: string;
}

// ── Panel grid (ports _panel_grid: ncols=min(n,2)) ───────────────────────────

export interface PanelBox {
  left: number; // percent
  top: number;
  width: number;
  height: number;
  index: number;
}

export function panelGrid(
  n: number,
  opts: { legendRows?: number; panelW?: number; panelH?: number } = {}
): { boxes: PanelBox[]; width: number; height: number; ncols: number; nrows: number } {
  const ncols = Math.min(n, 2);
  const nrows = Math.ceil(n / ncols);
  const panelW = opts.panelW ?? 640;
  const panelH = opts.panelH ?? 440;
  const legendPad = 6 + (opts.legendRows ?? 1) * 5; // percent reserved at bottom
  const width = panelW * ncols;
  const height = panelH * nrows + 90;
  const boxes: PanelBox[] = [];
  const availH = 100 - legendPad - 6;
  for (let i = 0; i < n; i++) {
    const col = i % ncols;
    const row = Math.floor(i / ncols);
    boxes.push({
      index: i,
      left: 7 + col * (93 / ncols),
      top: 8 + row * (availH / nrows),
      width: 93 / ncols - 9,
      height: availH / nrows - 12,
    });
  }
  return { boxes, width, height, ncols, nrows };
}

// ── Theme-aware axis/text defaults ───────────────────────────────────────────

export function baseTextStyle(theme: GenTheme, size = 14) {
  return { color: theme.fg, fontSize: size };
}

export function axisStyle(theme: GenTheme, opts: { name?: string; nameSize?: number } = {}) {
  return {
    name: opts.name,
    nameTextStyle: { color: theme.axisLabelColor, fontSize: opts.nameSize ?? 16, fontWeight: "bold" as const },
    nameLocation: "middle" as const,
    nameGap: 30,
    axisLine: { lineStyle: { color: theme.axisLabelColor } },
    axisLabel: { color: theme.axisLabelColor, fontSize: 13 },
    splitLine: { lineStyle: { color: theme.gridColor, opacity: theme.gridAlpha } },
  };
}

export function panelTitle(text: string, box: PanelBox, theme: GenTheme, size = 17) {
  return {
    text,
    left: `${box.left + box.width / 2}%`,
    top: `${Math.max(box.top - 5, 0.5)}%`,
    textAlign: "center" as const,
    textStyle: { color: theme.fg, fontSize: size, fontWeight: "bold" as const },
  };
}

// ── symlog transform helpers (matplotlib symlog(linthresh=1) equivalent) ─────

export function symlogVals(values: number[], on: boolean): number[] {
  return on ? values.map((v) => symlog(v)) : values;
}

export function symlogAxisLabelFormatter(on: boolean): ((v: number) => string) | undefined {
  if (!on) return undefined;
  return (v: number) => {
    const raw = v === 0 ? 0 : Math.sign(v) * (Math.abs(v) <= 1 ? Math.abs(v) : 10 ** (Math.abs(v) - 1));
    if (Math.abs(raw) >= 1000) return `${Math.round(raw / 1000)}k`;
    if (Math.abs(raw) >= 10) return String(Math.round(raw));
    return raw.toFixed(1).replace(/\.0$/, "");
  };
}

// ── Error bar custom series (mean ± lo/hi whiskers) ─────────────────────────

export interface ErrorBarPoint {
  x: number | string;
  low: number;
  high: number;
}

export function errorBarSeries(
  points: ErrorBarPoint[],
  color: string,
  opts: { xAxisIndex?: number; yAxisIndex?: number } = {}
): SeriesOption {
  return {
    type: "custom",
    silent: true,
    z: 10,
    xAxisIndex: opts.xAxisIndex ?? 0,
    yAxisIndex: opts.yAxisIndex ?? 0,
    data: points.map((p) => [p.x, p.low, p.high]),
    renderItem: (_params, api) => {
      const x = api.value(0) as number;
      const lo = api.coord([x, api.value(1)]);
      const hi = api.coord([x, api.value(2)]);
      const cap = 4;
      const line = { stroke: color, lineWidth: 1.1 };
      return {
        type: "group",
        children: [
          { type: "line", shape: { x1: hi[0], y1: hi[1], x2: lo[0], y2: lo[1] }, style: line },
          { type: "line", shape: { x1: hi[0] - cap, y1: hi[1], x2: hi[0] + cap, y2: hi[1] }, style: line },
          { type: "line", shape: { x1: lo[0] - cap, y1: lo[1], x2: lo[0] + cap, y2: lo[1] }, style: line },
        ],
      };
    },
  } as SeriesOption;
}

/** Pareto step line points (ports the step-front construction). */
export function stepFront(points: [number, number][]): [number, number][] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (!pts.length) return [];
  const out: [number, number][] = [pts[0]];
  for (let j = 1; j < pts.length; j++) {
    out.push([pts[j][0], pts[j - 1][1]], [pts[j][0], pts[j][1]]);
  }
  return out;
}

/**
 * Empty named series backing decorative legend entries: ECharts only renders
 * legend items whose name matches a series (or data) name, so purely visual
 * legends (variant colours, marker shapes, front lines) need placeholders.
 */
export function legendPlaceholderSeries(
  legendData: ReadonlyArray<{ name: string } | string>
): SeriesOption[] {
  return legendData.map((item) => ({
    name: typeof item === "string" ? item : item.name,
    type: "scatter" as const,
    data: [],
    silent: true,
  }));
}

// ── Marker/hatch cycles (ports MARKER_CYCLE / HATCH_CYCLE) ───────────────────

export const MARKER_CYCLE = [
  "circle",
  "rect",
  "diamond",
  "triangle",
  "arrow",
  "pin",
  "roundRect",
  "path://M0,0L10,10M10,0L0,10",
] as const;

/** ECharts decal patterns approximating the matplotlib hatch cycle ("", //, xx, ..). */
export const HATCH_DECALS: (object | undefined)[] = [
  undefined,
  { symbol: "line", rotation: Math.PI / 4, dashArrayX: [1, 0], dashArrayY: [3, 3], color: "rgba(0,0,0,0.35)" },
  { symbol: "line", rotation: -Math.PI / 4, dashArrayX: [1, 0], dashArrayY: [2, 2], color: "rgba(0,0,0,0.35)" },
  { symbol: "circle", symbolSize: 0.6, dashArrayX: [4, 4], dashArrayY: [4, 4], color: "rgba(0,0,0,0.35)" },
];

// ── RdYlGn colormap (matplotlib parity for heatmaps) ─────────────────────────

const RDYLGN: [number, string][] = [
  [0.0, "#a50026"],
  [0.15, "#d73027"],
  [0.3, "#f46d43"],
  [0.45, "#fdae61"],
  [0.5, "#ffffbf"],
  [0.6, "#d9ef8b"],
  [0.75, "#a6d96a"],
  [0.9, "#66bd63"],
  [1.0, "#1a9850"],
];

export function rdYlGnColors(reversed: boolean): string[] {
  const colors = RDYLGN.map(([, c]) => c);
  return reversed ? [...colors].reverse() : colors;
}

// ── Headless rendering (preview ≡ export by construction) ────────────────────

/** Render an option off-screen and return a PNG data URL. */
export async function renderChartPng(spec: ChartSpec, pixelRatio = 2): Promise<string> {
  const host = document.createElement("div");
  host.style.cssText = `position:fixed;left:-100000px;top:0;width:${spec.width}px;height:${spec.height}px;`;
  document.body.appendChild(host);
  const chart = echarts.init(host, undefined, {
    renderer: "canvas",
    width: spec.width,
    height: spec.height,
  });
  try {
    chart.setOption({ animation: false, ...spec.option });
    // one frame so custom series finish layout
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    return chart.getDataURL({
      type: "png",
      pixelRatio,
      backgroundColor: spec.background ?? (spec.option.backgroundColor as string) ?? "#ffffff",
    });
  } finally {
    chart.dispose();
    host.remove();
  }
}
