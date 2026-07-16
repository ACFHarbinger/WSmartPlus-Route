/**
 * Self-contained HTML report export (§H.5) — converts a generated analysis
 * markdown report into one standalone, styled HTML document with every
 * referenced figure inlined as a data URL. Links to the §H.5 interactive
 * chart pages are preserved (they are themselves self-contained files).
 */
import { marked } from "marked";
import { joinPath, pathExists, readBinaryDataUrl, readTextFile, writeTextFile } from "../io";
import type { Progress } from "./simulationReport";

function dirname(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i >= 0 ? path.slice(0, i) : "";
}

const REPORT_CSS = `
  :root { color-scheme: light; }
  body { margin: 0 auto; max-width: 980px; padding: 2.5em 1.5em 4em; background: #ffffff; color: #1F2D3D;
         font-family: Helvetica, Arial, sans-serif; line-height: 1.55; }
  h1 { font-size: 1.9em; border-bottom: 3px solid #2E74B5; padding-bottom: 0.3em; }
  h2 { font-size: 1.45em; color: #1F2D3D; border-bottom: 1px solid #C8D0DA; padding-bottom: 0.2em; margin-top: 2em; }
  h3 { font-size: 1.15em; margin-top: 1.6em; }
  h4 { font-size: 1.0em; margin-top: 1.3em; }
  a { color: #2E74B5; }
  blockquote { border-left: 4px solid #2E74B5; margin: 1em 0; padding: 0.3em 1em; background: #F0F4FA; color: #5A6A7A; }
  blockquote p { margin: 0.3em 0; }
  code { background: #F0F4FA; border-radius: 4px; padding: 0.1em 0.35em; font-size: 0.9em; }
  figure { margin: 1.2em 0; }
  figure img, p img { max-width: 100%; height: auto; border: 1px solid #E2E8F0; border-radius: 6px; }
  table { border-collapse: collapse; display: block; overflow-x: auto; max-width: 100%; font-size: 0.82em; margin: 1em 0; }
  th, td { border: 1px solid #C8D0DA; padding: 0.35em 0.6em; text-align: center; white-space: nowrap; }
  th { background: #1F2D3D; color: #ffffff; }
  tbody tr:nth-child(even) { background: #EEF2F7; }
  hr { border: none; border-top: 1px solid #C8D0DA; margin: 2em 0; }
  .toc-note, em { color: #5A6A7A; }
`;

/** Build the standalone HTML page for a written markdown report (§H.5/§H.7). */
export async function buildHtmlReportPage(
  projectRoot: string,
  mdRel: string,
  progress: Progress = () => {}
): Promise<string> {
  const mdAbs = joinPath(projectRoot, mdRel);
  progress(`Converting ${mdRel} → HTML …`);
  const md = await readTextFile(mdAbs);
  let html = marked.parse(md, { gfm: true, async: false }) as string;

  // Inline every referenced image (report-relative paths resolve against the
  // markdown file's directory, matching how the markdown is browsed).
  const baseDir = dirname(mdAbs);
  const srcs = new Set<string>();
  for (const m of html.matchAll(/<img[^>]+src="([^"]+)"/g)) {
    if (!/^(data:|https?:)/.test(m[1])) srcs.add(m[1]);
  }
  let inlined = 0;
  for (const src of srcs) {
    const abs = joinPath(baseDir, src);
    if (!(await pathExists(abs))) {
      progress(`[WARN] Referenced figure missing: ${src}`);
      continue;
    }
    const dataUrl = await readBinaryDataUrl(abs);
    html = html.split(`src="${src}"`).join(`src="${dataUrl}"`);
    inlined++;
  }
  progress(`Inlined ${inlined}/${srcs.size} figures`);

  const titleMatch = /<h1[^>]*>(.*?)<\/h1>/.exec(html);
  const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g, "") : "Analysis Report";
  const page = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
<style>${REPORT_CSS}</style>
</head>
<body>
${html}
</body>
</html>
`;
  return page;
}

/**
 * Convert a written markdown report into a standalone HTML file beside it.
 * Returns the project-relative output path.
 */
export async function generateHtmlReport(
  projectRoot: string,
  mdRel: string,
  progress: Progress = () => {}
): Promise<string> {
  const page = await buildHtmlReportPage(projectRoot, mdRel, progress);
  const outRel = mdRel.replace(/\.md$/i, "") + ".html";
  await writeTextFile(joinPath(projectRoot, outRel), page);
  progress(`Written: ${outRel} (${(page.length / 1e6).toFixed(1)} MB)`);
  return outRel;
}
