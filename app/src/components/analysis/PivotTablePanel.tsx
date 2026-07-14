/**
 * Lightweight pivot table for SQL/CSV grids (§G.6).
 */
import { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  buildPivot,
  pivotHeatmapOption,
  type PivotAgg,
} from "../../utils/pivotTable";

interface Props {
  columns: string[];
  rows: Record<string, unknown>[];
  onRowClick?: (rowKey: string, rowLabel: string) => void;
  /** Bidirectional brush: dim pivot rows not matching active policy filter (§G.6). */
  highlightRowLabels?: string[] | null;
}

export function PivotTablePanel({ columns, rows, onRowClick, highlightRowLabels }: Props) {
  const [rowKey, setRowKey] = useState(columns[0] ?? "");
  const [colKey, setColKey] = useState<string>("");
  const [valueKey, setValueKey] = useState(columns.find((c) => c !== rowKey) ?? "");
  const [agg, setAgg] = useState<PivotAgg>("mean");

  const numericCols = useMemo(
    () =>
      columns.filter((c) =>
        rows.every((r) => {
          const v = r[c];
          return v == null || v === "" || !Number.isNaN(Number(v));
        })
      ),
    [columns, rows]
  );

  const pivot = useMemo(() => {
    if (!rowKey || !valueKey) return null;
    return buildPivot(rows, {
      rowKey,
      colKey: colKey || null,
      valueKey,
      agg,
    });
  }, [rows, rowKey, colKey, valueKey, agg]);

  const pivotHighlights = useMemo(() => {
    if (!highlightRowLabels?.length || !/^policy$/i.test(rowKey)) return null;
    return highlightRowLabels;
  }, [highlightRowLabels, rowKey]);

  const chartOption = useMemo(
    () =>
      pivot
        ? pivotHeatmapOption(
            pivot,
            `${agg}(${valueKey}) by ${rowKey}${colKey ? ` × ${colKey}` : ""}`,
            pivotHighlights
          )
        : null,
    [pivot, agg, valueKey, rowKey, colKey, pivotHighlights]
  );

  if (columns.length < 2 || rows.length < 2) return null;

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-gray-300">Pivot Table (§G.6)</p>
      <div className="flex flex-wrap gap-2 text-xs">
        <label className="flex items-center gap-1">
          <span className="text-canvas-muted">Rows</span>
          <select
            className="select-base text-xs py-0.5"
            value={rowKey}
            onChange={(e) => setRowKey(e.target.value)}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <span className="text-canvas-muted">Columns</span>
          <select
            className="select-base text-xs py-0.5"
            value={colKey}
            onChange={(e) => setColKey(e.target.value)}
          >
            <option value="">— none —</option>
            {columns.filter((c) => c !== rowKey).map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <span className="text-canvas-muted">Value</span>
          <select
            className="select-base text-xs py-0.5"
            value={valueKey}
            onChange={(e) => setValueKey(e.target.value)}
          >
            {numericCols.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <span className="text-canvas-muted">Agg</span>
          <select
            className="select-base text-xs py-0.5"
            value={agg}
            onChange={(e) => setAgg(e.target.value as PivotAgg)}
          >
            <option value="mean">mean</option>
            <option value="sum">sum</option>
            <option value="count">count</option>
          </select>
        </label>
      </div>

      {chartOption && (
        <ReactECharts
          option={chartOption}
          style={{ height: Math.max(160, (pivot?.rowLabels.length ?? 0) * 22 + 80) }}
          onEvents={
            onRowClick && rowKey
              ? {
                  click: (params: { componentType?: string; value?: [number, number, number] }) => {
                    if (params.componentType !== "series" || !params.value || !pivot) return;
                    const ri = params.value[1];
                    const label = pivot.rowLabels[ri];
                    if (label) onRowClick(rowKey, label);
                  },
                }
              : undefined
          }
        />
      )}

      {onRowClick && /policy/i.test(rowKey) && (
        <p className="text-[10px] text-canvas-muted">
          Click a pivot row to cross-filter analytics views
          {pivotHighlights?.length ? " · highlighted rows match active filter" : ""}.
        </p>
      )}
    </div>
  );
}
