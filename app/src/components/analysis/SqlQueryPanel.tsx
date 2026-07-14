/**
 * DuckDB-Wasm SQL query editor panel (§G.6).
 */
import { lazy, Suspense, useCallback, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChevronDown, ChevronUp, Download, Play } from "lucide-react";
import { toast } from "sonner";
import { PivotTablePanel } from "./PivotTablePanel";
import { useGlobalFiltersStore } from "../../store/filters";
import { queryDuckDb } from "../../utils/duckdbClient";
import { sqlTemplates } from "../../utils/duckdbTemplates";
import { buildAutoChartOption, suggestChart } from "../../utils/queryAutoChart";
import { downloadCsv } from "../../utils/tableExport";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

interface Props {
  tableName: string;
  theme: "dark" | "light";
}

export function SqlQueryPanel({ tableName, theme }: Props) {
  const setPolicy = useGlobalFiltersStore((s) => s.setPolicy);
  const [open, setOpen] = useState(false);
  const [sql, setSql] = useState(`SELECT * FROM "${tableName}" LIMIT 100`);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const templates = useMemo(() => sqlTemplates(tableName), [tableName]);

  const columns = useMemo(() => {
    if (!rows.length) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  const chartSpec = useMemo(
    () => (columns.length ? suggestChart(columns, rows) : null),
    [columns, rows]
  );

  const chartOption = useMemo(
    () => (chartSpec ? buildAutoChartOption(chartSpec, rows) : null),
    [chartSpec, rows]
  );

  const sortedRows = useMemo(() => {
    if (!sortCol) return rows;
    return [...rows].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = typeof av === "number" ? av : Number(av);
      const bn = typeof bv === "number" ? bv : Number(bv);
      const bothNumeric = !Number.isNaN(an) && !Number.isNaN(bn);
      const cmp = bothNumeric
        ? an - bn
        : String(av ?? "").localeCompare(String(bv ?? ""));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [rows, sortCol, sortDir]);

  const runQuery = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await queryDuckDb<Record<string, unknown>>(sql);
      setRows(result);
      setSortCol(null);
    } catch (err) {
      setError(String(err));
      setRows([]);
    } finally {
      setRunning(false);
    }
  }, [sql]);

  const exportResults = useCallback(() => {
    if (!columns.length) return;
    downloadCsv(
      "duckdb-query.csv",
      columns,
      sortedRows.map((row) => columns.map((c) => String(row[c] ?? "")))
    );
  }, [columns, sortedRows]);

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

  const applyCrossFilter = (col: string, value: string) => {
    if (/^policy$/i.test(col)) {
      setPolicy(value);
      toast.success("Cross-filter applied", { description: `Policy: ${value}` });
    }
  };

  const handlePivotCrossFilter = (rowKey: string, label: string) => {
    applyCrossFilter(rowKey, label);
  };

  return (
    <div className="card space-y-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-xs font-semibold text-gray-300">DuckDB SQL Explorer (§G.6)</p>
          <p className="text-[10px] text-canvas-muted">Table: {tableName}</p>
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <>
          <div className="flex flex-wrap gap-1.5">
            {templates.map((tpl) => (
              <button
                key={tpl.id}
                onClick={() => setSql(tpl.sql)}
                className="btn-ghost text-xs"
              >
                {tpl.label}
              </button>
            ))}
          </div>

          <Suspense
            fallback={<p className="text-xs text-canvas-muted py-8 text-center">Loading editor…</p>}
          >
            <div className="rounded-lg border border-canvas-border overflow-hidden">
              <MonacoEditor
                height="140px"
                language="sql"
                theme={theme === "dark" ? "vs-dark" : "light"}
                value={sql}
                onChange={(v) => setSql(v ?? "")}
                options={{
                  minimap: { enabled: false },
                  fontSize: 11,
                  lineNumbers: "off",
                  wordWrap: "on",
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
              />
            </div>
          </Suspense>

          <div className="flex items-center gap-2">
            <button
              onClick={() => void runQuery()}
              disabled={running}
              className="btn-primary text-xs flex items-center gap-1"
            >
              <Play size={12} />
              {running ? "Running…" : "Run query"}
            </button>
            {rows.length > 0 && (
              <button onClick={exportResults} className="btn-ghost text-xs flex items-center gap-1">
                <Download size={12} />
                Export CSV
              </button>
            )}
            {rows.length > 0 && (
              <span className="text-xs text-canvas-muted">{rows.length} row(s)</span>
            )}
          </div>

          {error && <p className="text-xs text-accent-danger font-mono">{error}</p>}

          {chartSpec && chartOption && (
            <div className="space-y-1">
              <p className="text-xs text-canvas-muted">
                Auto-chart (§G.6): <span className="text-gray-300">{chartSpec.label}</span>
              </p>
              <ReactECharts option={chartOption} style={{ height: 200 }} />
            </div>
          )}

          {sortedRows.length > 0 && (
            <PivotTablePanel
              columns={columns}
              rows={rows}
              onRowClick={handlePivotCrossFilter}
            />
          )}

          {sortedRows.length > 0 && (
            <div className="overflow-auto max-h-64 rounded-lg border border-canvas-border">
              <table className="w-full text-xs">
                <thead className="bg-canvas-elevated sticky top-0">
                  <tr>
                    {columns.map((c) => (
                      <th
                        key={c}
                        onClick={() => toggleSort(c)}
                        className="px-3 py-2 text-left text-canvas-muted font-medium whitespace-nowrap cursor-pointer hover:text-gray-200"
                      >
                        <span className="inline-flex items-center gap-1">
                          {c}
                          {sortCol === c &&
                            (sortDir === "asc" ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-canvas-border">
                  {sortedRows.slice(0, 200).map((row, i) => (
                    <tr
                      key={i}
                      className="hover:bg-canvas-hover cursor-pointer"
                      onClick={() => {
                        const policyCol = columns.find((c) => /^policy$/i.test(c));
                        if (policyCol && row[policyCol] != null) {
                          applyCrossFilter(policyCol, String(row[policyCol]));
                        }
                      }}
                    >
                      {columns.map((c) => (
                        <td key={c} className="px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono">
                          {String(row[c] ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
