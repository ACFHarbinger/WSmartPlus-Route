/**
 * Headless browser environment for the native §H generation pipeline:
 * jsdom + node-canvas for ECharts rasterisation, resvg for SVG→PNG, and a
 * fetch shim that serves vite asset URLs from disk.
 *
 * Import (and await setup) BEFORE importing any module under `src/gen`.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import { Resvg } from "@resvg/resvg-js";

export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

export function setupHeadlessDom(): void {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost/",
    pretendToBeVisual: true,
    resources: "usable",
  });
  const win = dom.window;
  const g = globalThis as Record<string, unknown>;
  g.window = win;
  for (const key of [
    "document",
    "Image",
    "HTMLElement",
    "HTMLCanvasElement",
    "HTMLImageElement",
    "HTMLDivElement",
    "SVGElement",
    "FileReader",
    "Blob",
    "DOMParser",
    "getComputedStyle",
    "requestAnimationFrame",
    "cancelAnimationFrame",
    "devicePixelRatio",
  ]) {
    try {
      g[key] = (win as unknown as Record<string, unknown>)[key];
    } catch {
      /* non-configurable global — leave the Node one */
    }
  }
  try {
    Object.defineProperty(globalThis, "navigator", { value: win.navigator, configurable: true });
  } catch {
    /* Node ≥21 exposes navigator; jsdom's is only needed for UA sniffing */
  }

  // Serve vite-style asset URLs (assetToDataUrl fetches its PNG/webp imports).
  const origFetch = globalThis.fetch?.bind(globalThis);
  g.fetch = async (input: unknown, init?: unknown) => {
    const url =
      typeof input === "string" ? input : ((input as { url?: string })?.url ?? String(input));
    if (url.startsWith("/") || url.startsWith("file://")) {
      let p = url.startsWith("file://") ? fileURLToPath(url) : url.split("?")[0];
      if (p.startsWith("/@fs/")) p = p.slice("/@fs".length);
      else if (!fs.existsSync(p)) {
        const cand = path.join(APP_ROOT, p.replace(/^\//, ""));
        if (fs.existsSync(cand)) p = cand;
      }
      const buf = fs.readFileSync(p);
      // Minimal Response: the pipeline only calls `.blob()` (jsdom Blob so
      // jsdom's FileReader accepts it).
      return {
        ok: true,
        status: 200,
        blob: async () => new win.Blob([new Uint8Array(buf)]),
        arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
        text: async () => buf.toString("utf8"),
      };
    }
    if (!origFetch) throw new Error(`headless fetch: no network fetch available for ${url}`);
    return origFetch(input as RequestInfo, init as RequestInit);
  };
}

/** SVG→PNG via resvg (node-canvas prebuilds cannot decode SVG images). */
export async function resvgRasterizer(
  svg: string,
  width: number,
  _height: number,
  scale: number
): Promise<string> {
  const resvg = new Resvg(svg, {
    fitTo: { mode: "width", value: Math.round(width * scale) },
    background: "#ffffff",
  });
  const png = resvg.render().asPng();
  return `data:image/png;base64,${Buffer.from(png).toString("base64")}`;
}
