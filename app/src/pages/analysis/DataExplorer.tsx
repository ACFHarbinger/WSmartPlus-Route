/**
 * Data Explorer — browse and inspect simulation output CSV files.
 * Ports Streamlit `data_explorer` mode.
 */
import { useCallback, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Download } from "lucide-react";
import { toast } from "sonner";
import { SqlQueryPanel } from "../../components/analysis/SqlQueryPanel";
import { useAppStore } from "../../store/app";
import { recentFileLabel, useRecentFilesStore } from "../../store/recentFiles";
import { downloadCsv, downloadParquetFromCsv } from "../../utils/tableExport";
import { runCsvArrowPipeline } from "../../utils/arrowPipeline";
import { useDuckDbStore } from "../../store/duckdb";

interface CsvRow {
  [key: string]: string | number | null;
}

interface CsvFile {
  path: string;
  headers: string[];
  rows: CsvRow[];
}

export function DataExplorer() {
  const { projectRoot, theme } = useAppStore();
  const pushRecent = useRecentFilesStore((s) => s.pushRecent);
  const { ready: duckdbReady, lastPipeline, setLastPipeline, setLoading, loading } =
    useDuckDbStore();
  const [file, setFile] = useState<CsvFile | null>(null);
  const [exporting, setExporting] = useState(false);
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [filterText, setFilterText] = useState("");
  const PAGE_SIZE = 50;

  const openCsv = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "CSV Files", extensions: ["csv"] }],
    })) as string | null;
    if (!path) return;
    const loaded = await invoke<CsvFile>("load_csv_file", { path });
    setFile(loaded);
    setPage(0);
    setSortCol(null);
    setSortDir("asc");
    setFilterText("");
    pushRecent({ path, label: recentFileLabel(path), kind: "csv" });

    if (duckdbReady) {
      setLoading(true);
      try {
        const timing = await runCsvArrowPipeline(path, "explorer_csv");
        setLastPipeline(timing);
      } catch (err) {
        console.warn("DuckDB ingest failed:", err);
      } finally {
        setLoading(false);
      }
    }
  }, [pushRecent, duckdbReady, setLastPipeline, setLoading]);

  const filteredRows = useMemo(() => {
    if (!file) return [];
    const q = filterText.trim().toLowerCase();
    if (!q) return file.rows;
    return file.rows.filter((row) =>
      file.headers.some((h) => String(row[h] ?? "").toLowerCase().includes(q))
    );
  }, [file, filterText]);

  const sortedRows = useMemo(() => {
    if (!file) return [];
    if (!sortCol) return filteredRows;
    const rows = [...filteredRows];
    rows.sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = typeof av === "number" ? av : Number(av);
      const bn = typeof bv === "number" ? bv : Number(bv);
      const bothNumeric = !Number.isNaN(an) && !Number.isNaN(bn) && av !== "" && bv !== "";
      const cmp = bothNumeric
        ? an - bn
        : String(av ?? "").localeCompare(String(bv ?? ""));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return rows;
  }, [file, filteredRows, sortCol, sortDir]);

  const pageRows = sortedRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = file ? Math.ceil(sortedRows.length / PAGE_SIZE) : 0;

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortCol(col);
      setSortDir("asc");
    }
    setPage(0);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={openCsv} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open CSV
        </button>
        {file && (
          <>
            <span className="text-xs text-canvas-muted">
              {file.path.split("/").pop()} · {file.rows.length.toLocaleString()} rows
              {loading && " · DuckDB ingesting…"}
              {!loading && lastPipeline?.tableName === "explorer_csv" && (
                <>
                  {" "}
                  · DuckDB {lastPipeline.rowCount} rows in {lastPipeline.totalMs} ms
                  {lastPipeline.withinBudget ? "" : " (over 500ms budget)"}
                </>
              )}
            </span>
            <button
              onClick={() =>
                downloadCsv(
                  file.path.split("/").pop() ?? "data.csv",
                  file.headers,
                  sortedRows.map((row) => file.headers.map((h) => row[h] ?? ""))
                )
              }
              className="btn-ghost text-xs flex items-center gap-1.5"
            >
              <Download size={12} />
              Export CSV
            </button>
            {projectRoot && (
              <button
                disabled={exporting}
                onClick={async () => {
                  setExporting(true);
                  try {
                    const out = await downloadParquetFromCsv(
                      projectRoot,
                      file.path,
                      file.path.replace(/\.csv$/i, ".parquet")
                    );
                    if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
                  } catch (err) {
                    toast.error("Parquet export failed", { description: String(err) });
                  } finally {
                    setExporting(false);
                  }
                }}
                className="btn-ghost text-xs flex items-center gap-1.5"
              >
                <Download size={12} />
                Export Parquet
              </button>
            )}
          </>
        )}
      </div>

      {!file && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Open a CSV file from the output directory to explore.
        </div>
      )}

      {file && duckdbReady && lastPipeline?.tableName && (
        <SqlQueryPanel tableName={lastPipeline.tableName} theme={theme} />
      )}

      {file && (
        <>
          <div className="flex items-center gap-3">
            <input
              type="search"
              className="input-base text-xs flex-1 max-w-sm"
              placeholder="Filter rows (matches any column)…"
              value={filterText}
              onChange={(e) => {
                setFilterText(e.target.value);
                setPage(0);
              }}
            />
            {filterText && (
              <span className="text-xs text-canvas-muted">
                {sortedRows.length.toLocaleString()} / {file.rows.length.toLocaleString()} rows
              </span>
            )}
          </div>
          <div className="overflow-auto rounded-xl border border-canvas-border">
            <table className="w-full text-xs">
              <thead className="bg-canvas-elevated sticky top-0">
                <tr>
                  {file.headers.map((h) => (
                    <th
                      key={h}
                      onClick={() => toggleSort(h)}
                      className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap cursor-pointer hover:text-gray-200 select-none"
                    >
                      <span className="inline-flex items-center gap-1">
                        {h}
                        {sortCol === h &&
                          (sortDir === "asc" ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-canvas-border">
                {pageRows.map((row, i) => (
                  <tr key={i} className="hover:bg-canvas-hover">
                    {file.headers.map((h) => (
                      <td key={h} className="px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono">
                        {row[h] ?? "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center gap-3 text-xs text-canvas-muted">
              <button
                className="btn-ghost py-1 px-2"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                ← Prev
              </button>
              <span>
                Page {page + 1} / {totalPages}
              </span>
              <button
                className="btn-ghost py-1 px-2"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
