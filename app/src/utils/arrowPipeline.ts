/**
 * CSV → Rust Arrow IPC → DuckDB-Wasm pipeline (§G.0 Phase 0).
 * Prefers pre-built ``.arrow`` sidecars from ``.wsroute`` bundles when present
 * (CSV and simulation JSONL; §G.8).
 */
import { invoke } from "@tauri-apps/api/core";
import { duckDbRowCount, ingestArrowIpc, initDuckDb } from "./duckdbClient";

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
  tableName = "studio_csv"
): Promise<ArrowPipelineTiming> {
  const sidecar = csvArrowSidecarPath(csvPath);
  const hasSidecar = await invoke<boolean>("path_exists", { path: sidecar });
  if (hasSidecar) {
    const timing = await runArrowSidecarPipeline(sidecar, tableName);
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
  tableName = "studio_sim"
): Promise<ArrowPipelineTiming> {
  const sidecar = jsonlArrowSidecarPath(logPath);
  const hasSidecar = await invoke<boolean>("path_exists", { path: sidecar });
  if (hasSidecar) {
    return runArrowSidecarPipeline(sidecar, tableName);
  }

  const t0 = performance.now();
  await initDuckDb();

  const ipc = await invoke<ArrowIpcMeta>("simulation_log_to_arrow_ipc", { path: logPath });
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
    usedSidecar: false,
  };
}
