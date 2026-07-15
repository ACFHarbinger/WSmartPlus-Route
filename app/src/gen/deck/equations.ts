/**
 * Equation pipeline (§H.4) — native LaTeX rendering for the deck.
 *
 * Replaces the Python pandoc→OMML path: LaTeX is rendered to SVG via MathJax
 * (offline, bundled) and embedded as a high-resolution image, with the ported
 * `_plain_fallback` text kept for speaker notes / alt text. Native editable
 * OMML embedding is the remaining §H.4 item (LaTeX→MathML→OMML in Rust).
 */
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";
import { svgToPngDataUrl } from "./svg";

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const texInput = new TeX({ packages: AllPackages });
const svgOutput = new SVG({ fontCache: "local" });
const mjDocument = mathjax.document("", { InputJax: texInput, OutputJax: svgOutput });

const EX_TO_PX = 8; // 1ex at the MathJax default 16px em

export interface RenderedEquation {
  dataUrl: string;
  /** CSS pixel size of the rendered equation at scale 1. */
  width: number;
  height: number;
  fallback: string;
}

/** Render one LaTeX line to a PNG data URL (transparent-safe white background). */
export async function renderEquationPng(
  latex: string,
  opts: { color?: string; scale?: number } = {}
): Promise<RenderedEquation> {
  const node = mjDocument.convert(latex, { display: true });
  let svg = adaptor.innerHTML(node);
  // resolve ex-based dimensions to px so the rasteriser has concrete sizes
  const wMatch = /width="([\d.]+)ex"/.exec(svg);
  const hMatch = /height="([\d.]+)ex"/.exec(svg);
  const width = wMatch ? parseFloat(wMatch[1]) * EX_TO_PX : 320;
  const height = hMatch ? parseFloat(hMatch[1]) * EX_TO_PX : 48;
  svg = svg
    .replace(/width="[\d.]+ex"/, `width="${width}px"`)
    .replace(/height="[\d.]+ex"/, `height="${height}px"`);
  if (opts.color) {
    svg = svg.replace("<svg ", `<svg color="${opts.color}" `);
  }
  const dataUrl = await svgToPngDataUrl(svg, width, height, opts.scale ?? 4);
  return { dataUrl, width, height, fallback: plainFallback(latex) };
}

// ── Plain-text fallback (ports _plain_fallback, symbol table kept as data) ───

const FALLBACK_SYMBOLS: [string, string][] = [
  ["\\Rightarrow", " => "],
  ["\\rightarrow", " -> "],
  ["\\geq", " >= "],
  ["\\leq", " <= "],
  ["\\in", " in "],
  ["\\notin", " not in "],
  ["\\times", " x "],
  ["\\cdot", " * "],
  ["\\quad", "   "],
  ["\\qquad", "     "],
];

const FALLBACK_DROP = ["mathbf", "mathrm", "textbf", "text", "mathcal", "left", "right", "dfrac", "frac"];

export function plainFallback(latex: string): string {
  let text = latex;
  for (const [sym, plain] of FALLBACK_SYMBOLS) text = text.split(sym).join(plain);
  text = text.replace(new RegExp(`\\\\(${FALLBACK_DROP.join("|")})\\b`, "g"), "");
  text = text.replace(/\\([a-zA-Z]+)/g, "$1"); // keep the word, drop the backslash
  text = text.replace(/[\\{}$]/g, "").replace(/,/g, " ");
  return text.replace(/\s+/g, " ").trim();
}
