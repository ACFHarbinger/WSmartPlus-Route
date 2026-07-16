/**
 * PDF deck export (§H.6) — renders the HTML deck slides in-app with the real
 * WebKit engine (`html-to-image`, SVG foreignObject — full CSS fidelity
 * including the chevron clip-paths) and assembles a 16:9 landscape PDF via
 * jsPDF. No Python, no print dialog: preview ≡ output by construction.
 */
import { jsPDF } from "jspdf";
import { toPng } from "html-to-image";
import { joinPath, writeBinaryFile } from "../io";
import { buildHtmlDeckSlides, DECK_CSS, type HtmlDeckOptions } from "./htmlDeck";
import type { Progress } from "../report/simulationReport";

const SLIDE_W_PX = 1600;
const SLIDE_H_PX = 900;

export async function generateDeckPdf(
  opts: HtmlDeckOptions,
  progress: Progress = () => {}
): Promise<string> {
  const slides = await buildHtmlDeckSlides(opts, progress);
  progress(`Rasterising ${slides.length} slides for PDF …`);

  // Off-screen host with the deck stylesheet; fixed pixel geometry overrides
  // the slideshow's viewport-relative sizing.
  const host = document.createElement("div");
  host.style.cssText = `position:fixed;left:-100000px;top:0;width:${SLIDE_W_PX}px;height:${SLIDE_H_PX}px;overflow:hidden;`;
  const style = document.createElement("style");
  style.textContent =
    DECK_CSS +
    `\n.pdf-frame { width: ${SLIDE_W_PX}px; height: ${SLIDE_H_PX}px; display: flex; }` +
    `\n.pdf-frame .slide { width: ${SLIDE_W_PX}px; height: ${SLIDE_H_PX}px; aspect-ratio: auto; font-size: 22px; }`;
  host.appendChild(style);
  const frame = document.createElement("div");
  frame.className = "pdf-frame";
  host.appendChild(frame);
  document.body.appendChild(host);

  const pdf = new jsPDF({
    orientation: "landscape",
    unit: "px",
    format: [SLIDE_W_PX, SLIDE_H_PX],
    compress: true,
  });

  try {
    for (let i = 0; i < slides.length; i++) {
      frame.innerHTML = slides[i].html;
      // allow embedded data-URL images to decode
      await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 30)));
      const png = await toPng(frame, {
        width: SLIDE_W_PX,
        height: SLIDE_H_PX,
        pixelRatio: 1.5,
        backgroundColor: "#ffffff",
      });
      if (i > 0) pdf.addPage([SLIDE_W_PX, SLIDE_H_PX], "landscape");
      pdf.addImage(png, "PNG", 0, 0, SLIDE_W_PX, SLIDE_H_PX);
      progress(`Rasterised slide ${i + 1}/${slides.length}`);
    }
  } finally {
    host.remove();
  }

  const buffer = pdf.output("arraybuffer");
  const arr = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < arr.length; i += 0x8000) {
    binary += String.fromCharCode(...arr.subarray(i, i + 0x8000));
  }
  const outPath = opts.out.replace(/\.html?$/i, "").replace(/\.pptx$/i, "") + ".pdf";
  await writeBinaryFile(joinPath(opts.projectRoot, outPath), btoa(binary));
  progress(`Written: ${outPath} (${slides.length} pages)`);
  return outPath;
}
