/**
 * CSV → Rust Arrow IPC → DuckDB-Wasm pipeline (§G.0 Phase 0).
 * Prefers pre-built ``.arrow`` sidecars from ``.wsroute`` bundles when present
 * (CSV and simulation JSONL; §G.8).
 */
import { invoke } from "@tauri-apps/api/core";
import { cityScaleFromRunLabel } from "./cityComparison";
import {
  duckDbHasColumn,
  duckDbRowCount,
  ingestArrowIpc,
  initDuckDb,
  queryDuckDb,
} from "./duckdbClient";
import { runLabelFromSourcePath } from "./policyTelemetryTrends";

export const ARROW_PIPELINE_BUDGET_MS = 500;

export interface ArrowIpcMeta {
  path: string;
  row_count: number;
  column_count: number;
  rust_ms: number;
}

export interface ArrowPipelineTiming {
  rowCount: number;
  columnCount: number;
  rustMs: number;
  readMs: number;
  duckdbMs: number;
  totalMs: number;
  withinBudget: boolean;
  tableName: string;
  /** True when a sibling ``.arrow`` sidecar was ingested instead of re-parsing CSV. */
  usedSidecar?: boolean;
  /** Portfolio ingest: number of logs combined into the table. */
  logCount?: number;
  /** Portfolio ingest: logs ingested via Arrow sidecar fast-path. */
  sidecarCount?: number;
}

function sqlStringLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

/** Add ``run_label`` + ``city_scale`` when a single-log table lacks portfolio columns (§G.6 / §D.7). */
async function annotateTableWithRunLabelIfMissing(
  tableName: string,
  sourcePath: string,
  projectRoot?: string | null
): Promise<void> {
  const hasRunLabel = await duckDbHasColumn(tableName, "run_label");
  if (hasRunLabel) return;

  const runLabel = runLabelFromSourcePath(sourcePath, projectRoot);
  const cityScale = cityScaleFromRunLabel(runLabel);
  const rawName = `_annotate_${tableName}_raw`;
  await queryDuckDb(`ALTER TABLE "${tableName}" RENAME TO "${rawName}"`);
  await queryDuckDb(
    `CREATE TABLE "${tableName}" AS SELECT *, ${sqlStringLiteral(runLabel)} AS run_label, ${sqlStringLiteral(cityScale)} AS city_scale FROM "${rawName}"`
  );
  await queryDuckDb(`DROP TABLE IF EXISTS "${rawName}"`).catch(() => {});
}

/** Human-readable DuckDB ingest timing for toolbar badges (§G.0 / §G.7). */
export function formatPipelineTimingBadge(timing: ArrowPipelineTiming): string {
  let badge = `DuckDB ${timing.rowCount} rows`;
  if (timing.logCount != null && timing.logCount > 1) {
    badge += ` (${timing.logCount} runs)`;
  }
  badge += ` in ${timing.totalMs} ms`;
  if (
    timing.sidecarCount != null &&
    timing.logCount != null &&
    timing.logCount > 1 &&
    timing.sidecarCount > 0
  ) {
    badge += ` (${timing.sidecarCount}/${timing.logCount} Arrow sidecars)`;
  } else if (timing.usedSidecar) {
    badge += " (Arrow sidecar)";
  }
  if (!timing.withinBudget) {
    badge += " (over 500ms budget)";
  }
  return badge;
}

async function resolveSimulationArrowBuffer(logPath: string): Promise<{
  buffer: Uint8Array;
  usedSidecar: boolean;
  rustMs: number;
  columnCount: number;
}> {
  const sidecar = jsonlArrowSidecarPath(logPath);
  const hasSidecar = await invoke<boolean>("path_exists", { path: sidecar });
  if (hasSidecar) {
    const buffer = await readIpcBytes(sidecar);
    return { buffer, usedSidecar: true, rustMs: 0, columnCount: 0 };
  }

  const ipc = await invoke<ArrowIpcMeta>("simulation_log_to_arrow_ipc", { path: logPath });
  const buffer = await readIpcBytes(ipc.path);
  return {
    buffer,
    usedSidecar: false,
    rustMs: ipc.rust_ms,
    columnCount: ipc.column_count,
  };
}

function toUint8Array(bytes: number[]): Uint8Array {
  return Uint8Array.from(bytes);
}

async function readIpcBytes(ipcPath: string): Promise<Uint8Array> {
  const bytes = await invoke<number[]>("read_binary_file", { path: ipcPath });
  return toUint8Array(bytes);
}

/** Resolve the Arrow IPC sidecar path for a CSV file. */
export function csvArrowSidecarPath(csvPath: string): string {
  return csvPath.replace(/\.csv$/i, ".arrow");
}

/** Resolve the Arrow IPC sidecar path for a simulation JSONL log. */
export function jsonlArrowSidecarPath(logPath: string): string {
  return logPath.replace(/\.jsonl$/i, ".arrow");
}

/** Ingest a pre-built Arrow IPC file directly into DuckDB-Wasm. */
export async function runArrowSidecarPipeline(
  arrowPath: string,
  tableName = "studio_arrow"
): Promise<ArrowPipelineTiming> {
  const t0 = performance.now();
  await initDuckDb();

  const buffer = await readIpcBytes(arrowPath);
  const t1 = performance.now();

  await ingestArrowIpc(tableName, buffer);
  const t2 = performance.now();

  const rowCount = await duckDbRowCount(tableName);
  const totalMs = Math.round(t2 - t0);

  return {
    rowCount,
    columnCount: 0,
    rustMs: 0,
    readMs: Math.round(t1 - t0),
    duckdbMs: Math.round(t2 - t1),
    totalMs,
    withinBudget: totalMs < ARROW_PIPELINE_BUDGET_MS,
    tableName,
    usedSidecar: true,
  };
}

/** Full pipeline for an on-disk CSV file; prefers a sibling ``.arrow`` sidecar when present. */
export async function runCsvArrowPipeline(
  csvPath: string,
  tableName = "studio_csv",
  projectRoot?: string | null
): Promise<ArrowPipelineTiming> {
  const sidecar = csvArrowSidecarPath(csvPath);
  const hasSidecar = await invoke<boolean>("path_exists", { path: sidecar });
  if (hasSidecar) {
    const timing = await runArrowSidecarPipeline(sidecar, tableName);
    await annotateTableWithRunLabelIfMissing(tableName, csvPath, projectRoot);
    return { ...timing, columnCount: timing.columnCount };
  }

  const t0 = performance.now();
  await initDuckDb();

  const ipc = await invoke<ArrowIpcMeta>("csv_to_arrow_ipc", { path: csvPath });
  const t1 = performance.now();

  const buffer = await readIpcBytes(ipc.path);
  const t2 = performance.now();

  await ingestArrowIpc(tableName, buffer);
  const t3 = performance.now();

  const rowCount = await duckDbRowCount(tableName);
  await annotateTableWithRunLabelIfMissing(tableName, csvPath, projectRoot);
  const totalMs = Math.round(performance.now() - t0);

  return {
    rowCount,
    columnCount: ipc.column_count,
    rustMs: ipc.rust_ms,
    readMs: Math.round(t2 - t1),
    duckdbMs: Math.round(t3 - t2),
    totalMs,
    withinBudget: totalMs < ARROW_PIPELINE_BUDGET_MS,
    tableName,
    usedSidecar: false,
  };
}

/** Tensor slice → Arrow IPC → DuckDB-Wasm (§G.5.1). */
export async function runTensorArrowPipeline(
  npzPath: string,
  opts: {
    key?: string | null;
    indices?: number[];
    maxDim?: number;
    tableName?: string;
    projectRoot?: string;
    pythonExecutable?: string;
  } = {}
): Promise<ArrowPipelineTiming> {
  const tableName = opts.tableName ?? "studio_tensor";
  const t0 = performance.now();
  await initDuckDb();

  const ipc = await invoke<ArrowIpcMeta>("tensor_slice_to_arrow_ipc", {
    path: npzPath,
    key: opts.key ?? null,
    indices: opts.indices ?? [],
    maxDim: opts.maxDim ?? 64,
    projectRoot: opts.projectRoot ?? null,
    pythonExecutable: opts.pythonExecutable || null,
  });
  const t1 = performance.now();

  const buffer = await readIpcBytes(ipc.path);
  const t2 = performance.now();

  await ingestArrowIpc(tableName, buffer);
  const t3 = performance.now();

  const rowCount = await duckDbRowCount(tableName);
  const totalMs = Math.round(t3 - t0);

  return {
    rowCount,
    columnCount: ipc.column_count,
    rustMs: ipc.rust_ms,
    readMs: Math.round(t2 - t1),
    duckdbMs: Math.round(t3 - t2),
    totalMs,
    withinBudget: totalMs < ARROW_PIPELINE_BUDGET_MS,
    tableName,
  };
}

/** Full pipeline for a simulation JSONL log; prefers a sibling ``.arrow`` sidecar when present. */
export async function runSimulationArrowPipeline(
  logPath: string,
  tableName = "studio_sim",
  projectRoot?: string | null
): Promise<ArrowPipelineTiming> {
  const sidecar = jsonlArrowSidecarPath(logPath);
  const hasSidecar = await invoke<boolean>("path_exists", { path: sidecar });
  if (hasSidecar) {
    const timing = await runArrowSidecarPipeline(sidecar, tableName);
    await annotateTableWithRunLabelIfMissing(tableName, logPath, projectRoot);
    return { ...timing, logCount: 1, sidecarCount: 1 };
  }

  const t0 = performance.now();
  await initDuckDb();

  const { buffer, rustMs, columnCount } = await resolveSimulationArrowBuffer(logPath);
  const t1 = performance.now();

  await ingestArrowIpc(tableName, buffer);
  const t2 = performance.now();

  const rowCount = await duckDbRowCount(tableName);
  await annotateTableWithRunLabelIfMissing(tableName, logPath, projectRoot);
  const totalMs = Math.round(performance.now() - t0);

  return {
    rowCount,
    columnCount,
    rustMs,
    readMs: Math.round(t1 - t0),
    duckdbMs: Math.round(t2 - t1),
    totalMs,
    withinBudget: totalMs < ARROW_PIPELINE_BUDGET_MS,
    tableName,
    usedSidecar: false,
    logCount: 1,
    sidecarCount: 0,
  };
}

/**
 * Portfolio pipeline: union multiple simulation JSONL logs into one DuckDB table with
 * a ``run_label`` column (§G.1.4 / §G.6).
 */
export async function runPortfolioSimulationArrowPipeline(
  logs: { path: string; label: string }[],
  tableName = "studio_portfolio"
): Promise<ArrowPipelineTiming> {
  if (logs.length === 0) {
    throw new Error("No simulation logs to ingest");
  }

  const t0 = performance.now();
  await initDuckDb();

  const tempTables: string[] = [];
  let totalRustMs = 0;
  let sidecarCount = 0;
  let maxCols = 0;

  for (let i = 0; i < logs.length; i++) {
    const tmpName = `_portfolio_${i}`;
    const { buffer, usedSidecar, rustMs, columnCount } = await resolveSimulationArrowBuffer(
      logs[i].path
    );
    await ingestArrowIpc(tmpName, buffer);
    tempTables.push(tmpName);
    totalRustMs += rustMs;
    if (usedSidecar) sidecarCount++;
    maxCols = Math.max(maxCols, columnCount);
  }

  const unionSql = logs
    .map((log, i) => {
      const cityScale = cityScaleFromRunLabel(log.label);
      return `SELECT *, ${sqlStringLiteral(log.label)} AS run_label, ${sqlStringLiteral(cityScale)} AS city_scale FROM "${tempTables[i]}"`;
    })
    .join("\nUNION ALL BY NAME\n");

  await queryDuckDb(`DROP TABLE IF EXISTS "${tableName}"`);
  await queryDuckDb(`CREATE TABLE "${tableName}" AS ${unionSql}`);

  for (const tmp of tempTables) {
    await queryDuckDb(`DROP TABLE IF EXISTS "${tmp}"`).catch(() => {});
  }

  const rowCount = await duckDbRowCount(tableName);
  const totalMs = Math.round(performance.now() - t0);

  return {
    rowCount,
    columnCount: maxCols,
    rustMs: totalRustMs,
    readMs: 0,
    duckdbMs: 0,
    totalMs,
    withinBudget: totalMs < ARROW_PIPELINE_BUDGET_MS,
    tableName,
    usedSidecar: sidecarCount === logs.length,
    logCount: logs.length,
    sidecarCount,
  };
}
