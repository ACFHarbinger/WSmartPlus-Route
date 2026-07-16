/**
 * Headless §H generation runner — produces the simulation analysis (figures,
 * markdown, interactive HTML) and/or the presentation deck using the Studio's
 * own `src/gen` code, without the GUI.
 *
 * Usage (from app/):
 *   npx vite-node --config scripts/headless/vite.config.ts scripts/gen-headless.ts -- \
 *     [--project-root <dir>] [--report] [--deck] [--out <pptx>] [--figures-dir <dir>] \
 *     [--results-table 30d|90d|all|none] [--html] [--pdf] [--speaker-script] [--excel] \
 *     [--no-interactive] [--force]
 *
 * With neither --report nor --deck, both run (analysis first, then the deck).
 */
import path from "node:path";
import { APP_ROOT, resvgRasterizer, setupHeadlessDom } from "./headless/env";

interface Cli {
  projectRoot: string;
  report: boolean;
  deck: boolean;
  out: string;
  figuresDir: string;
  resultsTable: "30d" | "90d" | "all" | "none";
  html: boolean;
  pdf: boolean;
  speakerScript: boolean;
  excel: boolean;
  interactive: boolean | undefined;
  force: boolean;
}

function parseCli(argv: string[]): Cli {
  const has = (f: string) => argv.includes(f);
  const val = (f: string): string | undefined => {
    const i = argv.indexOf(f);
    return i !== -1 ? argv[i + 1] : undefined;
  };
  const report = has("--report");
  const deck = has("--deck");
  return {
    projectRoot: path.resolve(val("--project-root") ?? path.join(APP_ROOT, "..")),
    report: report || !deck,
    deck: deck || !report,
    out: val("--out") ?? "assets/windows/wsmart_route_results.pptx",
    figuresDir: val("--figures-dir") ?? "public/figures/simulation/30d",
    resultsTable: (val("--results-table") as Cli["resultsTable"]) ?? "30d",
    html: has("--html"),
    pdf: has("--pdf"),
    speakerScript: has("--speaker-script"),
    excel: has("--excel"),
    interactive: has("--no-interactive") ? false : undefined,
    force: has("--force"),
  };
}

async function main(): Promise<void> {
  const cli = parseCli(process.argv.slice(2));
  setupHeadlessDom();

  // Import gen modules only after the DOM globals exist.
  const { setSvgRasterizer } = await import("../src/gen/deck/svg");
  setSvgRasterizer(resvgRasterizer);

  const progress = (msg: string) => console.log(msg);
  const outputs: string[] = [];

  if (cli.report) {
    console.log(`── simulation analysis (project root: ${cli.projectRoot}) ──`);
    const { generateSimulationReport } = await import("../src/gen/report/simulationReport");
    const res = await generateSimulationReport(
      {
        projectRoot: cli.projectRoot,
        force: cli.force,
        interactive: cli.interactive,
      },
      progress
    );
    if (res.outMd) outputs.push(res.outMd);
    outputs.push(...res.figuresDirs);
  }

  if (cli.deck) {
    console.log(`── presentation deck ──`);
    const { generatePresentation } = await import("../src/gen/deck/deckBuilder");
    const res = await generatePresentation(
      {
        projectRoot: cli.projectRoot,
        figuresDir: cli.figuresDir,
        out: cli.out,
        resultsTable: cli.resultsTable,
        resultsTableSplit: "none",
        imageMode: "native",
        speakerScript: cli.speakerScript,
        excel: cli.excel,
        html: cli.html,
        pdf: cli.pdf,
      },
      progress
    );
    outputs.push(...res.outputs);
  }

  console.log("\nOutputs:");
  for (const o of outputs) console.log(`  ${o}`);
}

// top-level await: vite-node tears down its module server as soon as the
// entry module finishes evaluating, so main() must be awaited here.
try {
  await main();
} catch (err) {
  console.error(err);
  process.exitCode = 1;
}
