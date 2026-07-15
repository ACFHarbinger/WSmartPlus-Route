/**
 * Standalone OLAP Data Cube Explorer (§G.6).
 *
 * Full-page DuckDB-Wasm SQL editor with table picker for all ingested datasets.
 */
import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Database, FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { SqlQueryPanel } from "../../components/analysis/SqlQueryPanel";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useAppStore } from "../../store/app";
import { useDuckDbStore } from "../../store/duckdb";
import { useGlobalFiltersStore } from "../../store/filters";
import {
  formatPipelineTimingBadge,
  runCsvArrowPipeline,
  runSimulationArrowPipeline,
} from "../../utils/arrowPipeline";
import {
  duckDbRowCount,
  listDuckDbDistinctValues,
  listDuckDbTables,
} from "../../utils/duckdbClient";

const CUSTOM_TABLE_PREFIX = "olap_";

const PORTFOLIO_TABLES = new Set([
  "summary_sim",
  "benchmark_sim",
  "city_sim",
  "algorithm_sim",
  "studio_portfolio",
]);

export function OlapExplorer() {
  const { theme } = useAppStore();
  const activePolicy = useGlobalFiltersStore((s) => s.policy);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const {
    ready: duckdbReady,
    loading,
    lastPipeline,
    setLoading,
    setLastPipeline,
  } = useDuckDbStore();
  const [tables, setTables] = useState<string[]>([]);
  const [selectedTable, setSelectedTable] = useState("summary_sim");
  const [rowCounts, setRowCounts] = useState<Record<string, number>>({});
  const [refreshing, setRefreshing] = useState(false);
  const [runLabels, setRunLabels] = useState<string[]>([]);

  const refreshTables = useCallback(async () => {
    if (!duckdbReady) return;
    setRefreshing(true);
    try {
      const names = await listDuckDbTables();
      setTables(names);
      const counts: Record<string, number> = {};
      await Promise.all(
        names.map(async (name) => {
          counts[name] = await duckDbRowCount(name);
        })
      );
      setRowCounts(counts);
      if (names.length && !names.includes(selectedTable)) {
        setSelectedTable(names[0]);
      }
    } catch (err) {
      toast.error("Failed to list DuckDB tables", { description: String(err) });
    } finally {
      setRefreshing(false);
    }
  }, [duckdbReady, selectedTable]);

  useEffect(() => {
    void refreshTables();
  }, [refreshTables]);

  useEffect(() => {
    if (!duckdbReady || !selectedTable || !PORTFOLIO_TABLES.has(selectedTable)) {
      setRunLabels([]);
      return;
    }
    void (async () => {
      try {
        const labels = await listDuckDbDistinctValues(selectedTable, "run_label");
        setRunLabels(labels);
      } catch {
        setRunLabels([]);
      }
    })();
  }, [duckdbReady, selectedTable]);

  const ingestData = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "CSV / JSONL", extensions: ["csv", "jsonl"] }],
    })) as string | null;
    if (!path) return;
    const base =
      path
        .split(/[/\\]/)
        .pop()
        ?.replace(/\.(csv|jsonl)$/i, "") ?? "data";
    const tableName = `${CUSTOM_TABLE_PREFIX}${base.replace(/[^a-zA-Z0-9_]/g, "_")}`;
    const isJsonl = path.toLowerCase().endsWith(".jsonl");
    setLoading(true);
    try {
      const timing = isJsonl
        ? await runSimulationArrowPipeline(path, tableName)
        : await runCsvArrowPipeline(path, tableName);
      setLastPipeline(timing);
      setSelectedTable(tableName);
      await refreshTables();
      const sidecarNote = timing.usedSidecar ? " (Arrow sidecar)" : "";
      toast.success(isJsonl ? "JSONL ingested" : "CSV ingested", {
        description: `${timing.rowCount} rows → ${tableName}${sidecarNote}`,
      });
    } catch (err) {
      toast.error("Ingest failed", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [refreshTables, setLastPipeline, setLoading]);

  const highlightPolicies = activePolicy ? [activePolicy] : null;
  const highlightRunLabels = activeRunLabel ? [activeRunLabel] : null;
  const portfolioMode = PORTFOLIO_TABLES.has(selectedTable);

  return (
    <div className="space-y-4">
      <GlobalFilterBar runLabels={portfolioMode ? runLabels : []} />

      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => void refreshTables()}
          disabled={!duckdbReady || refreshing}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <RefreshCw size={12} className={refreshing ? "animate-spin" : ""} />
          Refresh tables
        </button>
        <button
          onClick={() => void ingestData()}
          disabled={!duckdbReady || loading}
          className="btn-primary text-xs flex items-center gap-1"
        >
          <FolderOpen size={12} />
          Ingest CSV / JSONL
        </button>
        <span className="text-xs text-canvas-muted flex items-center gap-1">
          <Database size={12} />
          {duckdbReady ? `${tables.length} table(s) in DuckDB-Wasm` : "DuckDB initialising…"}
          {loading && " · ingesting…"}
          {!loading && lastPipeline && <> · last ingest {formatPipelineTimingBadge(lastPipeline)}</>}
        </span>
      </div>

      {tables.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-gray-300 mb-2">Ingested tables</p>
          <div className="flex flex-wrap gap-2">
            {tables.map((name) => (
              <button
                key={name}
                onClick={() => setSelectedTable(name)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  selectedTable === name
                    ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                    : "border-canvas-border text-canvas-muted hover:text-gray-200"
                }`}
              >
                {name}
                <span className="ml-1.5 text-canvas-muted font-mono">
                  ({rowCounts[name] ?? "…"})
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {duckdbReady && selectedTable && (
        <SqlQueryPanel
          tableName={selectedTable}
          theme={theme}
          highlightPolicies={highlightPolicies}
          highlightRunLabels={highlightRunLabels}
          brushSqlSync
          autoRunOnBrushSync
          portfolioMode={portfolioMode}
          algorithmMode={selectedTable === "algorithm_sim"}
          defaultOpen
        />
      )}

      {!duckdbReady && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Waiting for DuckDB-Wasm worker…
        </div>
      )}

      {duckdbReady && tables.length === 0 && (
        <div className="card text-sm text-canvas-muted space-y-2">
          <p>
            No tables ingested yet. Load a simulation log in Simulation Summary, open a CSV in
            Data Explorer, or ingest a CSV / JSONL above (prefers sibling ``.arrow`` sidecars from
            ``.wsroute`` bundles).
          </p>
        </div>
      )}
    </div>
  );
}
