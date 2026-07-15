/**
 * Lightweight pivot table for SQL/CSV grids (§G.6).
 */
import { useCallback, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import {
  buildPivot,
  pivotHeatmapOption,
  type PivotAgg,
} from "../../utils/pivotTable";

interface Props {
  columns: string[];
  rows: Record<string, unknown>[];
  logScale?: boolean;
  onRowClick?: (rowKey: string, rowLabel: string) => void;
  /** Bidirectional brush: dim pivot rows not matching active policy filter (§G.6). */
  highlightPolicyLabels?: string[] | null;
  /** Bidirectional brush: dim pivot rows not matching active run_label filter (§G.6). */
  highlightRunLabels?: string[] | null;
  /** Bidirectional brush: dim pivot rows not matching active city/scale filter (§G.6). */
  highlightCityScaleLabels?: string[] | null;
}

type PivotWell = "row" | "col" | "value";

function DimensionWell({
  label,
  well,
  value,
  onDrop,
  onClear,
}: {
  label: string;
  well: PivotWell;
  value: string;
  onDrop: (well: PivotWell, col: string) => void;
  onClear: (well: PivotWell) => void;
}) {
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const onDropHandler = (e: React.DragEvent) => {
    e.preventDefault();
    const col = e.dataTransfer.getData("text/pivot-col");
    if (col) onDrop(well, col);
  };

  return (
    <div
      onDragOver={onDragOver}
      onDrop={onDropHandler}
      className="flex items-center gap-1 min-w-[120px] px-2 py-1.5 rounded-lg border border-dashed border-canvas-border bg-canvas-elevated/50"
    >
      <span className="text-[10px] text-canvas-muted uppercase">{label}</span>
      {value ? (
        <span className="flex items-center gap-1 text-xs text-gray-200 font-mono">
          {value}
          <button
            type="button"
            onClick={() => onClear(well)}
            className="text-canvas-muted hover:text-accent-danger text-[10px]"
            aria-label={`Clear ${label}`}
          >
            ×
          </button>
        </span>
      ) : (
        <span className="text-[10px] text-canvas-muted italic">drop column</span>
      )}
    </div>
  );
}

export function PivotTablePanel({
  columns,
  rows,
  logScale = false,
  onRowClick,
  highlightPolicyLabels,
  highlightRunLabels,
  highlightCityScaleLabels,
}: Props) {
  const chartRef = useRef<EChartsReact | null>(null);
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

  const handleWellDrop = useCallback(
    (well: PivotWell, col: string) => {
      if (well === "row") {
        setRowKey(col);
        if (colKey === col) setColKey("");
        if (valueKey === col) setValueKey(numericCols.find((c) => c !== col) ?? "");
      } else if (well === "col") {
        setColKey(col);
        if (rowKey === col) setRowKey(columns.find((c) => c !== col) ?? col);
      } else {
        setValueKey(col);
      }
    },
    [colKey, columns, numericCols, rowKey, valueKey]
  );

  const clearWell = useCallback((well: PivotWell) => {
    if (well === "row") setRowKey("");
    else if (well === "col") setColKey("");
    else setValueKey("");
  }, []);

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
    if (/^policy$/i.test(rowKey) && highlightPolicyLabels?.length) {
      return highlightPolicyLabels;
    }
    if (/^run_label$/i.test(rowKey) && highlightRunLabels?.length) {
      return highlightRunLabels;
    }
    if (/^city_scale$/i.test(rowKey) && highlightCityScaleLabels?.length) {
      return highlightCityScaleLabels;
    }
    return null;
  }, [highlightPolicyLabels, highlightRunLabels, highlightCityScaleLabels, rowKey]);

  const chartOption = useMemo(
    () =>
      pivot
        ? pivotHeatmapOption(
            pivot,
            `${agg}(${valueKey}) by ${rowKey}${colKey ? ` × ${colKey}` : ""}${logScale ? " · log" : ""}`,
            pivotHighlights,
            { logScale, valueKey }
          )
        : null,
    [pivot, agg, valueKey, rowKey, colKey, pivotHighlights, logScale]
  );

  if (columns.length < 2 || rows.length < 2) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-gray-300">Pivot Table (§G.6)</p>
        {chartOption && (
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem="pivot-heatmap"
            size={11}
            className="shrink-0"
          />
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {columns.map((c) => (
          <span
            key={c}
            draggable
            onDragStart={(e) => e.dataTransfer.setData("text/pivot-col", c)}
            className="text-[10px] px-2 py-0.5 rounded-md bg-canvas-elevated border border-canvas-border text-gray-300 cursor-grab active:cursor-grabbing"
          >
            {c}
          </span>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <DimensionWell
          label="Rows"
          well="row"
          value={rowKey}
          onDrop={handleWellDrop}
          onClear={clearWell}
        />
        <DimensionWell
          label="Columns"
          well="col"
          value={colKey}
          onDrop={handleWellDrop}
          onClear={clearWell}
        />
        <DimensionWell
          label="Value"
          well="value"
          value={valueKey}
          onDrop={handleWellDrop}
          onClear={clearWell}
        />
        <label className="flex items-center gap-1 text-xs self-center">
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
          ref={chartRef}
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

      {onRowClick && (/policy/i.test(rowKey) || /run_label/i.test(rowKey) || /city_scale/i.test(rowKey)) && (
        <p className="text-[10px] text-canvas-muted">
          Drag columns into wells · click pivot row to cross-filter
          {pivotHighlights?.length ? " · highlighted rows match active filter" : ""}.
        </p>
      )}
    </div>
  );
}
