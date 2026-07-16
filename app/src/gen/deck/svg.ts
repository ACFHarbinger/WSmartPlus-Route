/**
 * Minimal SVG toolkit for procedural deck illustrations (§H.3).
 *
 * Builds SVG strings and rasterises them to PNG data URLs via canvas so the
 * deck exporter can embed them exactly as previewed.
 */

export class SvgCanvas {
  private parts: string[] = [];
  constructor(
    public readonly width: number,
    public readonly height: number,
    background = "#ffffff"
  ) {
    this.parts.push(
      `<rect x="0" y="0" width="${width}" height="${height}" fill="${background}"/>`,
      `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker></defs>`
    );
  }

  roundRect(
    x: number,
    y: number,
    w: number,
    h: number,
    opts: { fill?: string; stroke?: string; strokeWidth?: number; rx?: number; dash?: string } = {}
  ): this {
    this.parts.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${opts.rx ?? 10}" ` +
        `fill="${opts.fill ?? "none"}" stroke="${opts.stroke ?? "none"}" ` +
        `stroke-width="${opts.strokeWidth ?? 1.5}"${opts.dash ? ` stroke-dasharray="${opts.dash}"` : ""}/>`
    );
    return this;
  }

  circle(cx: number, cy: number, r: number, opts: { fill?: string; stroke?: string; strokeWidth?: number; opacity?: number } = {}): this {
    this.parts.push(
      `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${opts.fill ?? "#000"}" ` +
        `stroke="${opts.stroke ?? "none"}" stroke-width="${opts.strokeWidth ?? 1.2}" opacity="${opts.opacity ?? 1}"/>`
    );
    return this;
  }

  rect(x: number, y: number, w: number, h: number, opts: { fill?: string; stroke?: string; strokeWidth?: number } = {}): this {
    this.parts.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${opts.fill ?? "#000"}" ` +
        `stroke="${opts.stroke ?? "none"}" stroke-width="${opts.strokeWidth ?? 1.2}"/>`
    );
    return this;
  }

  line(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    opts: { stroke?: string; width?: number; dash?: string; arrow?: boolean; opacity?: number } = {}
  ): this {
    this.parts.push(
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${opts.stroke ?? "#000"}" ` +
        `stroke-width="${opts.width ?? 1.5}"${opts.dash ? ` stroke-dasharray="${opts.dash}"` : ""}` +
        `${opts.arrow ? ` marker-end="url(#arrow)"` : ""} opacity="${opts.opacity ?? 1}"/>`
    );
    return this;
  }

  /** Quadratic-curved arrow (approximates matplotlib arc3 connection style). */
  curve(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    bend: number,
    opts: { stroke?: string; width?: number; arrow?: boolean } = {}
  ): this {
    const mx = (x1 + x2) / 2 - (y2 - y1) * bend;
    const my = (y1 + y2) / 2 + (x2 - x1) * bend;
    this.parts.push(
      `<path d="M${x1},${y1} Q${mx},${my} ${x2},${y2}" fill="none" stroke="${opts.stroke ?? "#000"}" ` +
        `stroke-width="${opts.width ?? 1.6}"${opts.arrow ? ` marker-end="url(#arrow)"` : ""}/>`
    );
    return this;
  }

  polyline(pts: [number, number][], opts: { stroke?: string; width?: number; fill?: string; dash?: string } = {}): this {
    this.parts.push(
      `<polyline points="${pts.map(([x, y]) => `${x},${y}`).join(" ")}" fill="${opts.fill ?? "none"}" ` +
        `stroke="${opts.stroke ?? "#000"}" stroke-width="${opts.width ?? 2}"${opts.dash ? ` stroke-dasharray="${opts.dash}"` : ""}/>`
    );
    return this;
  }

  polygon(pts: [number, number][], opts: { fill?: string; stroke?: string; strokeWidth?: number } = {}): this {
    this.parts.push(
      `<polygon points="${pts.map(([x, y]) => `${x},${y}`).join(" ")}" fill="${opts.fill ?? "#000"}" ` +
        `stroke="${opts.stroke ?? "none"}" stroke-width="${opts.strokeWidth ?? 1.2}"/>`
    );
    return this;
  }

  text(
    x: number,
    y: number,
    content: string,
    opts: {
      size?: number;
      color?: string;
      bold?: boolean;
      italic?: boolean;
      anchor?: "start" | "middle" | "end";
      lineHeight?: number;
    } = {}
  ): this {
    const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const lines = content.split("\n");
    const lh = opts.lineHeight ?? (opts.size ?? 12) * 1.25;
    const y0 = y - ((lines.length - 1) * lh) / 2;
    const style =
      `font-family="Helvetica,Arial,sans-serif" font-size="${opts.size ?? 12}" fill="${opts.color ?? "#000"}"` +
      `${opts.bold ? ' font-weight="bold"' : ""}${opts.italic ? ' font-style="italic"' : ""}`;
    for (let i = 0; i < lines.length; i++) {
      this.parts.push(
        `<text x="${x}" y="${y0 + i * lh}" text-anchor="${opts.anchor ?? "middle"}" dominant-baseline="middle" ${style}>${esc(lines[i])}</text>`
      );
    }
    return this;
  }

  raw(fragment: string): this {
    this.parts.push(fragment);
    return this;
  }

  toString(): string {
    return (
      `<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}" ` +
      `viewBox="0 0 ${this.width} ${this.height}">${this.parts.join("")}</svg>`
    );
  }
}

export type SvgRasterizer = (
  svg: string,
  width: number,
  height: number,
  scale: number
) => Promise<string>;

let rasterizerOverride: SvgRasterizer | null = null;

/** Inject a non-browser rasteriser (headless generation runs outside the webview). */
export function setSvgRasterizer(fn: SvgRasterizer | null): void {
  rasterizerOverride = fn;
}

/** Rasterise an SVG string to a PNG data URL. */
export async function svgToPngDataUrl(svg: string, width: number, height: number, scale = 2): Promise<string> {
  if (rasterizerOverride) return rasterizerOverride(svg, width, height, scale);
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("SVG rasterisation failed"));
    img.src = url;
  });
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

/** Deterministic PRNG (mulberry32) for seeded procedural illustrations. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller standard normal sampler over a PRNG. */
export function normal(rng: () => number): () => number {
  return () => {
    const u = Math.max(rng(), 1e-12);
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}
