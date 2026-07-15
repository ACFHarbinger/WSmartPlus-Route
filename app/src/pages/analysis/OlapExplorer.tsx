/**
 * Standalone OLAP Data Cube Explorer (§G.6).
 *
 * Full-page DuckDB-Wasm SQL editor with table picker for all ingested datasets.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { Database, FolderOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import {
  isSimulationLogPath,
  LogHandoffButtons,
} from "../../components/common/LogHandoffButtons";
import { PathRunLabelChip } from "../../components/common/PathRunLabelChip";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/SqlQueryPanel";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useLogPathRunLabelBrush } from "../../hooks/useLogPathRunLabelBrush";
import { useTableRunLabelBrush } from "../../hooks/useTableRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/useRecentHandoff";
import { useDuckDbStore } from "../../store/duckdb";

import { useGlobalFiltersStore } from "../../store/filters";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runCsvArrowPipeline,
  runPortfolioSimulationArrowPipeline,
} from "../../utils/arrowPipeline";
import { groupRunLabelsByCity } from "../../utils/cityComparison";
import {
  duckDbHasColumn,
  duckDbRowCount,
  listDuckDbDistinctValues,
  listDuckDbTables,
} from "../../utils/duckdbClient";
import {
  runLabelMapFromSingleTableLabels,
  runLabelMapFromTablePaths,
  tableRunLabelBrushActive,
} from "../../utils/policyTelemetryTrends";

const CUSTOM_TABLE_PREFIX = "olap_";

export function OlapExplorer() {
  const { effectiveTheme: theme } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const activePolicy = useGlobalFiltersStore((s) => s.policy);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const logScale = useGlobalFiltersStore((s) => s.logScale);
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
  const [policies, setPolicies] = useState<string[]>([]);
  const [cities, setCities] = useState<string[]>([]);
  const [portfolioMode, setPortfolioMode] = useState(false);
  const [ingestedTablePaths, setIngestedTablePaths] = useState<Record<string, string>>({});
  const [tableRunLabelsByName, setTableRunLabelsByName] = useState<Record<string, string[]>>({});

  const selectedIngestPath = ingestedTablePaths[selectedTable] ?? null;
  const selectedTableRunLabels = tableRunLabelsByName[selectedTable] ?? [];
  const derivedRunLabel = useLogPathRunLabelBrush(selectedIngestPath);
  const derivedTableRunLabel = useTableRunLabelBrush(
    selectedTable,
    selectedTableRunLabels,
    Boolean(selectedIngestPath)
  );
  const sourceRunLabel = useMemo(
    () =>
      selectedIngestPath
        ? portfolioRunLabel(selectedIngestPath, undefined, projectRoot)
        : null,
    [selectedIngestPath, projectRoot]
  );

  const tableBrushByName = useMemo(
    () => ({
      ...runLabelMapFromSingleTableLabels(tableRunLabelsByName),
      ...runLabelMapFromTablePaths(ingestedTablePaths, projectRoot),
    }),
    [tableRunLabelsByName, ingestedTablePaths, projectRoot]
  );

  const handleRunLabelClick = useCallback(
    (label: string) => {
      setRunLabel(activeRunLabel === label ? null : label);
    },
    [activeRunLabel, setRunLabel]
  );

  const refreshTables = useCallback(async () => {
    if (!duckdbReady) return;
    setRefreshing(true);
    try {
      const names = await listDuckDbTables();
      setTables(names);
      const counts: Record<string, number> = {};
      const labelsByTable: Record<string, string[]> = {};
      await Promise.all(
        names.map(async (name) => {
          counts[name] = await duckDbRowCount(name);
          const hasRunLabel = await duckDbHasColumn(name, "run_label");
          if (hasRunLabel) {
            labelsByTable[name] = await listDuckDbDistinctValues(name, "run_label");
          }
        })
      );
      setRowCounts(counts);
      setTableRunLabelsByName(labelsByTable);
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
    if (!duckdbReady || !selectedTable) {
      setRunLabels([]);
      setPolicies([]);
      setCities([]);
      setPortfolioMode(false);
      return;
    }
    void (async () => {
      try {
        const hasRunLabel = await duckDbHasColumn(selectedTable, "run_label");
        setPortfolioMode(hasRunLabel);
        const labels = hasRunLabel
          ? await listDuckDbDistinctValues(selectedTable, "run_label")
          : [];
        setRunLabels(labels);

        const hasPolicy = await duckDbHasColumn(selectedTable, "policy");
        setPolicies(
          hasPolicy ? await listDuckDbDistinctValues(selectedTable, "policy") : []
        );

        const hasCityScale = await duckDbHasColumn(selectedTable, "city_scale");
        if (hasCityScale) {
          const cityValues = await listDuckDbDistinctValues(selectedTable, "city_scale");
          setCities(cityValues.length > 1 ? cityValues : []);
        } else if (hasRunLabel) {
          const cityGroups = groupRunLabelsByCity(labels);
          setCities(cityGroups.length > 1 ? cityGroups.map(([city]) => city) : []);
        } else {
          setCities([]);
        }
      } catch {
        setRunLabels([]);
        setPolicies([]);
        setCities([]);
        setPortfolioMode(false);
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
    const runLabel = portfolioRunLabel(path, undefined, projectRoot);
    handoff(path, isJsonl ? "log" : "csv", { storedLabel: runLabel, navigate: false });
    setLoading(true);
    try {
      const timing = isJsonl
        ? await runPortfolioSimulationArrowPipeline(
            [{ path, label: runLabel }],
            tableName,
            projectRoot
          )
        : await runCsvArrowPipeline(path, tableName, projectRoot);
      setLastPipeline(timing);
      setSelectedTable(tableName);
      setIngestedTablePaths((prev) => ({ ...prev, [tableName]: path }));
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
  }, [projectRoot, refreshTables, setLastPipeline, setLoading, handoff]);

  const filterBarRunLabels = useMemo(() => {
    if (portfolioMode && runLabels.length > 0) return runLabels;
    if (sourceRunLabel) return [sourceRunLabel];
    if (derivedRunLabel) return [derivedRunLabel];
    if (derivedTableRunLabel) return [derivedTableRunLabel];
    return [];
  }, [portfolioMode, runLabels, sourceRunLabel, derivedRunLabel, derivedTableRunLabel]);

  const highlightPolicies = activePolicy ? [activePolicy] : null;

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        policies={policies}
        runLabels={filterBarRunLabels}
        cities={cities}
        showLogScale
      />

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
        {selectedIngestPath && isSimulationLogPath(selectedIngestPath) && (
          <LogHandoffButtons
            path={selectedIngestPath}
            storedLabel={sourceRunLabel ?? undefined}
            labeled
            iconSize={12}
          />
        )}
        {selectedIngestPath && (
          <PathRunLabelChip
            path={selectedIngestPath}
            projectRoot={projectRoot}
            trailing={
              !loading && lastPipeline ? (
                <span className="shrink-0">· {formatPipelineTimingBadge(lastPipeline)}</span>
              ) : undefined
            }
          />
        )}
        <span className="text-xs text-canvas-muted flex items-center gap-1">
          <Database size={12} />
          {duckdbReady ? `${tables.length} table(s) in DuckDB-Wasm` : "DuckDB initialising…"}
          {loading && " · ingesting…"}
          {!loading && lastPipeline && !selectedIngestPath && (
            <> · last ingest {formatPipelineTimingBadge(lastPipeline)}</>
          )}
        </span>
      </div>

      {tables.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-gray-300 mb-2">Ingested tables</p>
          <div className="flex flex-wrap gap-2">
            {tables.map((name) => {
              const runBrushActive =
                tableBrushByName[name] === activeRunLabel ||
                tableRunLabelBrushActive(tableRunLabelsByName[name], activeRunLabel);
              const tableRunLabel = tableBrushByName[name];
              return (
                <button
                  key={name}
                  onClick={() => {
                    setSelectedTable(name);
                    if (tableRunLabel) handleRunLabelClick(tableRunLabel);
                  }}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                    selectedTable === name
                      ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                      : "border-canvas-border text-canvas-muted hover:text-gray-200"
                  } ${runBrushActive ? "ring-1 ring-accent-secondary/40" : ""}`}
                >
                  {name}
                  <span className="ml-1.5 text-canvas-muted font-mono">
                    ({rowCounts[name] ?? "…"})
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {duckdbReady && (
        <PolicyTelemetryTrendsPanel
          theme={theme}
          logScale={logScale}
          initialPolicy={activePolicy}
          initialRunLabel={
            activeRunLabel ??
            (runLabels.length === 1 ? runLabels[0]! : null) ??
            sourceRunLabel ??
            derivedRunLabel ??
            derivedTableRunLabel
          }
        />
      )}

      {duckdbReady && selectedTable && (
        <SqlQueryPanel
          tableName={selectedTable}
          theme={theme}
          highlightPolicies={highlightPolicies}
          highlightRunLabels={
            portfolioMode && runLabels.length > 0
              ? null
              : sourceRunLabel
                ? [sourceRunLabel]
                : derivedTableRunLabel
                  ? [derivedTableRunLabel]
                  : null
          }
          brushSqlSync={
            portfolioMode || Boolean(sourceRunLabel || derivedRunLabel || derivedTableRunLabel)
          }
          autoRunOnBrushSync={
            portfolioMode || Boolean(sourceRunLabel || derivedRunLabel || derivedTableRunLabel)
          }
          portfolioMode={
            portfolioMode || Boolean(sourceRunLabel || derivedTableRunLabel)
          }
          algorithmMode={selectedTable === "algorithm_sim"}
          portfolioRunLabels={
            portfolioMode && runLabels.length > 0
              ? runLabels
              : sourceRunLabel
                ? [sourceRunLabel]
                : derivedTableRunLabel
                  ? [derivedTableRunLabel]
                  : []
          }
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
