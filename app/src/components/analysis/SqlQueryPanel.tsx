/**
 * DuckDB-Wasm SQL query editor panel (§G.6).
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChevronDown, ChevronUp, Download, ImageDown, Play } from "lucide-react";
import { toast } from "sonner";
import { PivotTablePanel } from "./PivotTablePanel";
import { useGlobalFiltersStore } from "../../store/filters";
import { resolveBrushedRunLabels } from "../../utils/cityComparison";
import { queryDuckDb } from "../../utils/duckdbClient";
import {
  brushedPortfolioSql,
  sqlTemplates,
  type PortfolioBrushFilter,
} from "../../utils/duckdbTemplates";
import { exportChartPngWithToast, exportChartSvgWithToast } from "../../utils/chartExport";
import {
  buildAutoChartOption,
  heatmapCellLabels,
  suggestChartAlternatives,
  type AutoChartSpec,
  type AutoChartType,
} from "../../utils/queryAutoChart";
import { downloadCsv } from "../../utils/tableExport";

const MonacoEditor = lazy(() => import("@monaco-editor/react"));

interface Props {
  tableName: string;
  theme: "dark" | "light";
  onDaySelect?: (day: number) => void;
  onProfitRange?: (min: number, max: number) => void;
  /** Multi-policy brush from chart panels (§G.1 / §G.6 bidirectional filter). */
  highlightPolicies?: string[] | null;
  /** Portfolio ``run_label`` brush from multi-run chart panels (§G.6). */
  highlightRunLabels?: string[] | null;
  /** Sync Monaco SQL to brushed-policies query when chart brush changes (§G.1). */
  brushSqlSync?: boolean;
  /** Auto-execute SQL when brush sync updates the editor (§G.2 segment → DuckDB). */
  autoRunOnBrushSync?: boolean;
  /** Panel starts expanded (e.g. when a chart brush is active). */
  defaultOpen?: boolean;
  /** Include portfolio ``run_label`` query templates (§G.6). */
  portfolioMode?: boolean;
  /** Include Algorithm Comparison policy-analysis templates (§G.6). */
  algorithmMode?: boolean;
  /** All portfolio ``run_label`` values — enables city brush SQL expansion (§G.6). */
  portfolioRunLabels?: string[];
}

export function SqlQueryPanel({
  tableName,
  theme,
  onDaySelect,
  onProfitRange,
  highlightPolicies: highlightPoliciesProp = null,
  highlightRunLabels: highlightRunLabelsProp = null,
  brushSqlSync = false,
  autoRunOnBrushSync = false,
  defaultOpen = false,
  portfolioMode = false,
  algorithmMode = false,
  portfolioRunLabels = [],
}: Props) {
  const activePolicy = useGlobalFiltersStore((s) => s.policy);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const brushedCity = useGlobalFiltersStore((s) => s.brushedCity);
  const setPolicy = useGlobalFiltersStore((s) => s.setPolicy);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const setBrushedCity = useGlobalFiltersStore((s) => s.setBrushedCity);
  const [open, setOpen] = useState(defaultOpen);
  const [sql, setSql] = useState(`SELECT * FROM "${tableName}" LIMIT 100`);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [filterText, setFilterText] = useState("");
  const [chartTypeOverride, setChartTypeOverride] = useState<AutoChartType | null>(null);
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const autoChartRef = useRef<ReactECharts>(null);

  const templates = useMemo(
    () => sqlTemplates(tableName, { portfolio: portfolioMode, algorithm: algorithmMode }),
    [tableName, portfolioMode, algorithmMode]
  );

  const brushFilter = useMemo((): PortfolioBrushFilter => {
    const filter: PortfolioBrushFilter = {};
    if (highlightPoliciesProp?.length) {
      filter.policies = highlightPoliciesProp;
    } else if (activePolicy) {
      filter.policies = [activePolicy];
    }
    if (highlightRunLabelsProp?.length) {
      filter.runLabels = highlightRunLabelsProp;
    } else if (brushedCity) {
      const expanded = resolveBrushedRunLabels(
        portfolioRunLabels,
        null,
        brushedCity
      );
      if (expanded?.length) {
        filter.runLabels = expanded;
      } else {
        filter.cityScale = brushedCity;
      }
    } else if (activeRunLabel) {
      filter.runLabels = [activeRunLabel];
    }
    return filter;
  }, [
    highlightPoliciesProp,
    highlightRunLabelsProp,
    activePolicy,
    activeRunLabel,
    brushedCity,
    portfolioRunLabels,
  ]);

  const hasBrushFilter = Boolean(
    brushFilter.policies?.length || brushFilter.runLabels?.length || brushFilter.cityScale
  );

  useEffect(() => {
    if (!brushSqlSync) return;
    setSql(brushedPortfolioSql(tableName, brushFilter));
    if (hasBrushFilter) setOpen(true);
  }, [brushSqlSync, brushFilter, hasBrushFilter, tableName]);

  useEffect(() => {
    if (!brushSqlSync || !autoRunOnBrushSync || !hasBrushFilter) return;
    const nextSql = brushedPortfolioSql(tableName, brushFilter);
    void (async () => {
      setRunning(true);
      setError(null);
      try {
        const result = await queryDuckDb<Record<string, unknown>>(nextSql);
        setRows(result);
        setSortCol(null);
        setFilterText("");
      } catch (err) {
        setError(String(err));
        setRows([]);
      } finally {
        setRunning(false);
      }
    })();
  }, [brushSqlSync, autoRunOnBrushSync, brushFilter, hasBrushFilter, tableName]);

  const columns = useMemo(() => {
    if (!rows.length) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  const chartAlternatives = useMemo(
    () => (columns.length ? suggestChartAlternatives(columns, rows) : []),
    [columns, rows]
  );

  const chartSpec = useMemo((): AutoChartSpec | null => {
    if (!chartAlternatives.length) return null;
    if (chartTypeOverride) {
      return chartAlternatives.find((s) => s.type === chartTypeOverride) ?? chartAlternatives[0];
    }
    return chartAlternatives[0];
  }, [chartAlternatives, chartTypeOverride]);

  useEffect(() => {
    setChartTypeOverride(null);
  }, [rows]);

  const chartOption = useMemo(
    () =>
      chartSpec
        ? buildAutoChartOption(chartSpec, rows, { logScale })
        : null,
    [chartSpec, rows, logScale]
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

  const filteredRows = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return sortedRows;
    return sortedRows.filter((row) =>
      columns.some((c) => String(row[c] ?? "").toLowerCase().includes(q))
    );
  }, [sortedRows, filterText, columns]);

  const runQuery = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await queryDuckDb<Record<string, unknown>>(sql);
      setRows(result);
      setSortCol(null);
      setFilterText("");
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
      filteredRows.map((row) => columns.map((c) => String(row[c] ?? "")))
    );
  }, [columns, filteredRows]);

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
      return;
    }
    if (/^run_label$/i.test(col)) {
      setBrushedCity(null);
      setRunLabel(value);
      toast.success("Cross-filter applied", { description: `Run: ${value}` });
      return;
    }
    if (/^city_scale$/i.test(col)) {
      setRunLabel(null);
      setBrushedCity(value);
      toast.success("Cross-filter applied", { description: `City: ${value}` });
    }
  };

  const profitCol = useMemo(
    () => columns.find((c) => /^profit$/i.test(c)) ?? null,
    [columns]
  );

  const policyCol = useMemo(
    () => columns.find((c) => /^policy$/i.test(c)) ?? null,
    [columns]
  );

  const runLabelCol = useMemo(
    () => columns.find((c) => /^run_label$/i.test(c)) ?? null,
    [columns]
  );

  const cityScaleCol = useMemo(
    () => columns.find((c) => /^city_scale$/i.test(c)) ?? null,
    [columns]
  );

  const highlightPolicies = useMemo(() => {
    if (highlightPoliciesProp?.length) return highlightPoliciesProp;
    if (activePolicy) return [activePolicy];
    return null;
  }, [highlightPoliciesProp, activePolicy]);

  const highlightRunLabels = useMemo(() => {
    if (highlightRunLabelsProp?.length) return highlightRunLabelsProp;
    const expanded = resolveBrushedRunLabels(
      portfolioRunLabels,
      activeRunLabel,
      brushedCity
    );
    if (expanded?.length) return expanded;
    if (activeRunLabel) return [activeRunLabel];
    return null;
  }, [highlightRunLabelsProp, portfolioRunLabels, activeRunLabel, brushedCity]);

  const highlightCityScale = useMemo(
    () => (brushedCity ? [brushedCity] : null),
    [brushedCity]
  );

  const rowMatchesHighlight = useCallback(
    (row: Record<string, unknown>) => {
      const policyMatch =
        !highlightPolicies?.length ||
        !policyCol ||
        highlightPolicies.includes(String(row[policyCol] ?? ""));
      const runMatch =
        !highlightRunLabels?.length ||
        !runLabelCol ||
        highlightRunLabels.includes(String(row[runLabelCol] ?? ""));
      const cityMatch =
        !brushedCity ||
        !cityScaleCol ||
        String(row[cityScaleCol] ?? "") === brushedCity;
      return policyMatch && runMatch && cityMatch;
    },
    [highlightPolicies, highlightRunLabels, brushedCity, policyCol, runLabelCol, cityScaleCol]
  );

  const applyProfitBrush = useCallback(() => {
    if (!profitCol || !rows.length || !onProfitRange) return;
    const vals = rows
      .map((r) => Number(r[profitCol]))
      .filter((v) => Number.isFinite(v));
    if (!vals.length) return;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    onProfitRange(min, max);
    toast.success("Profit brush applied", {
      description: `€${min.toFixed(0)} – €${max.toFixed(0)}`,
    });
  }, [profitCol, rows, onProfitRange]);

  const handlePivotCrossFilter = (rowKey: string, label: string) => {
    applyCrossFilter(rowKey, label);
  };

  const onAutoChartClick = useCallback(
    (params: {
      componentType?: string;
      name?: string;
      seriesName?: string;
      value?: unknown;
    }) => {
      if (!chartSpec || params.componentType !== "series") return;

      if (chartSpec.type === "bar") {
        applyCrossFilter(chartSpec.xKey, String(params.name ?? ""));
        return;
      }

      if (chartSpec.type === "grouped-bar" && chartSpec.seriesKey) {
        if (params.seriesName) {
          applyCrossFilter(chartSpec.seriesKey, String(params.seriesName));
        }
        if (params.name) {
          applyCrossFilter(chartSpec.xKey, String(params.name));
        }
        return;
      }

      if (chartSpec.type === "heatmap" && chartSpec.seriesKey) {
        const val = params.value as [number, number, number] | undefined;
        if (!val) return;
        const { colLabel, rowLabel } = heatmapCellLabels(
          chartSpec,
          rows,
          val[0],
          val[1]
        );
        if (colLabel) applyCrossFilter(chartSpec.xKey, colLabel);
        if (rowLabel) applyCrossFilter(chartSpec.seriesKey, rowLabel);
        return;
      }

      if (chartSpec.type === "scatter" && chartSpec.labelKey) {
        if (params.seriesName === "Pareto front") return;
        const label =
          params.name ??
          (typeof params.value === "object" &&
          params.value != null &&
          "name" in (params.value as object)
            ? String((params.value as { name?: string }).name ?? "")
            : "");
        if (label) applyCrossFilter(chartSpec.labelKey, label);
        return;
      }

      if (chartSpec.type === "line") {
        const raw =
          typeof params.value === "object" &&
          params.value != null &&
          Array.isArray(params.value)
            ? params.value[0]
            : params.name;
        const value = String(raw ?? "");
        if (/^(policy|run_label|city_scale)$/i.test(chartSpec.xKey)) {
          applyCrossFilter(chartSpec.xKey, value);
          return;
        }
        if (/^day$/i.test(chartSpec.xKey) && onDaySelect != null) {
          const day = Number(value);
          if (Number.isFinite(day)) onDaySelect(day);
        }
      }
    },
    [chartSpec, rows, applyCrossFilter, onDaySelect]
  );

  const autoChartCrossFilterHint = useMemo(() => {
    if (!chartSpec) return false;
    if (chartSpec.type === "bar") {
      return /^(policy|run_label|city_scale)$/i.test(chartSpec.xKey);
    }
    if (chartSpec.type === "grouped-bar" || chartSpec.type === "heatmap") {
      return Boolean(chartSpec.seriesKey);
    }
    if (chartSpec.type === "scatter" && chartSpec.labelKey) {
      return /^(policy|run_label|city_scale)$/i.test(chartSpec.labelKey);
    }
    if (chartSpec.type === "line") {
      return (
        /^(policy|run_label|city_scale|day)$/i.test(chartSpec.xKey) &&
        (onDaySelect != null || !/^day$/i.test(chartSpec.xKey))
      );
    }
    return false;
  }, [chartSpec, onDaySelect]);

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
              <span className="text-xs text-canvas-muted">
                {filterText.trim()
                  ? `${filteredRows.length.toLocaleString()} / ${rows.length.toLocaleString()} row(s)`
                  : `${rows.length.toLocaleString()} row(s)`}
              </span>
            )}
            {profitCol && onProfitRange && rows.length > 0 && (
              <button onClick={applyProfitBrush} className="btn-ghost text-xs">
                Brush profit range
              </button>
            )}
          </div>

          {error && <p className="text-xs text-accent-danger font-mono">{error}</p>}

          {chartSpec && chartOption && (
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-xs text-canvas-muted">
                  Auto-chart (§G.6): <span className="text-gray-300">{chartSpec.label}</span>
                </p>
                {chartAlternatives.length > 1 && (
                  <div className="flex flex-wrap gap-1">
                    {chartAlternatives.map((alt) => (
                      <button
                        key={alt.type}
                        type="button"
                        onClick={() => setChartTypeOverride(alt.type)}
                        className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                          chartSpec.type === alt.type
                            ? "border-accent-primary bg-accent-primary/15 text-accent-secondary"
                            : "border-canvas-border text-canvas-muted hover:text-gray-200"
                        }`}
                      >
                        {alt.type}
                      </button>
                    ))}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => exportChartPngWithToast(autoChartRef, "auto-chart.png")}
                  className="btn-ghost text-[10px] flex items-center gap-1 py-0.5"
                >
                  <ImageDown size={11} />
                  PNG
                </button>
                <button
                  type="button"
                  onClick={() => exportChartSvgWithToast(autoChartRef, "auto-chart.svg")}
                  className="btn-ghost text-[10px] flex items-center gap-1 py-0.5"
                >
                  <ImageDown size={11} />
                  SVG
                </button>

              </div>
              <ReactECharts
                ref={autoChartRef}
                option={chartOption}
                style={{ height: 200 }}
                onEvents={
                  autoChartCrossFilterHint ? { click: onAutoChartClick } : undefined
                }
              />
              {autoChartCrossFilterHint && (
                <p className="text-[10px] text-canvas-muted">
                  Click a bar, line point, scatter point, or heatmap cell to cross-filter linked
                  panels.
                </p>
              )}
            </div>
          )}

          {sortedRows.length > 0 && (
            <PivotTablePanel
              columns={columns}
              rows={rows}
              logScale={logScale}
              onRowClick={handlePivotCrossFilter}
              highlightPolicyLabels={highlightPolicies}
              highlightRunLabels={highlightRunLabels}
              highlightCityScaleLabels={highlightCityScale}
            />
          )}

          {sortedRows.length > 0 && (
            <input
              type="search"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter rows…"
              className="input-base w-full max-w-xs text-xs"
            />
          )}

          {filteredRows.length > 0 && (
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
                  {filteredRows.slice(0, 200).map((row, i) => {
                    const highlighted = rowMatchesHighlight(row);
                    const isActive =
                      (policyCol &&
                        activePolicy &&
                        String(row[policyCol] ?? "") === activePolicy) ||
                      (runLabelCol &&
                        activeRunLabel &&
                        String(row[runLabelCol] ?? "") === activeRunLabel) ||
                      (cityScaleCol &&
                        brushedCity &&
                        String(row[cityScaleCol] ?? "") === brushedCity);
                    const dayCol = columns.find((c) => /^day$/i.test(c));
                    return (
                    <tr
                      key={i}
                      className={`hover:bg-canvas-hover ${
                        isActive
                          ? "bg-accent-primary/15 ring-1 ring-inset ring-accent-primary/40"
                          : highlighted
                            ? ""
                            : "opacity-35"
                      }`}
                    >
                      {columns.map((c) => {
                        const isBrushCol =
                          c === policyCol || c === runLabelCol || c === cityScaleCol;
                        const isDayCol = c === dayCol && onDaySelect != null;
                        return (
                          <td
                            key={c}
                            onClick={() => {
                              if (isBrushCol) {
                                applyCrossFilter(c, String(row[c] ?? ""));
                                return;
                              }
                              if (isDayCol && row[c] != null) {
                                const day = Number(row[c]);
                                if (Number.isFinite(day)) onDaySelect(day);
                              }
                            }}
                            className={`px-3 py-1.5 text-gray-300 whitespace-nowrap font-mono ${
                              isBrushCol || isDayCol
                                ? "cursor-pointer hover:text-accent-secondary"
                                : ""
                            }`}
                          >
                            {String(row[c] ?? "—")}
                          </td>
                        );
                      })}
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
