/**
 * DuckDB-Wasm singleton (§G.0 Phase 0).
 * Runs in a dedicated Web Worker via the official duckdb-wasm bundle.
 */
import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";

const MANUAL_BUNDLES: duckdb.DuckDBBundles = {
  mvp: {
    mainModule: duckdb_wasm,
    mainWorker: duckdb_worker,
  },
};

let db: duckdb.AsyncDuckDB | null = null;
let conn: duckdb.AsyncDuckDBConnection | null = null;
let initPromise: Promise<void> | null = null;

export async function initDuckDb(): Promise<void> {
  if (conn) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
    const worker = new Worker(bundle.mainWorker!);
    const logger = new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING);
    db = new duckdb.AsyncDuckDB(logger, worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    conn = await db.connect();
  })();

  return initPromise;
}

export function isDuckDbReady(): boolean {
  return conn != null;
}

/** Register an Arrow IPC file buffer as a queryable table. */
export async function ingestArrowIpc(
  tableName: string,
  buffer: Uint8Array
): Promise<void> {
  await initDuckDb();
  if (!conn) throw new Error("DuckDB connection unavailable");
  await conn.query(`DROP TABLE IF EXISTS "${tableName}"`).catch(() => {});
  await conn.insertArrowFromIPCStream(buffer, { name: tableName });
}

export async function queryDuckDb<T extends Record<string, unknown>>(
  sql: string
): Promise<T[]> {
  await initDuckDb();
  if (!conn) throw new Error("DuckDB connection unavailable");
  const result = await conn.query(sql);
  return result.toArray() as T[];
}

export async function duckDbRowCount(tableName: string): Promise<number> {
  const rows = await queryDuckDb<{ cnt: number }>(
    `SELECT COUNT(*)::INTEGER AS cnt FROM "${tableName}"`
  );
  return rows[0]?.cnt ?? 0;
}

/** List ingested DuckDB tables (§G.6 standalone OLAP). */
export async function listDuckDbTables(): Promise<string[]> {
  const rows = await queryDuckDb<{ name: string }>("SHOW TABLES");
  return rows.map((r) => r.name).sort();
}
