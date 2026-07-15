/**
 * Cross-run policy telemetry trends from ``assets/telemetry.db`` (§A.3 Option C).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Database, RefreshCw } from "lucide-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { useAppStore } from "../../store/app";
import type { PolicyTelemetryTrends, PolicyVizType } from "../../types";
import {
  buildTrendComparisonOption,
  buildTrendStepsOption,
  formatTrendMetric,
  policyVizTypeLabel,
} from "../../utils/policyTelemetryTrends";

interface Props {
  theme: "dark" | "light";
  logScale?: boolean;
  /** Bump to reload trends after live telemetry emission. */
  refreshKey?: number;
}

export function PolicyTelemetryTrendsPanel({
  theme,
  logScale = false,
  refreshKey = 0,
}: Props) {
  const { projectRoot, pythonPath } = useAppStore();
  const compareRef = useRef<EChartsReact | null>(null);
  const stepsRef = useRef<EChartsReact | null>(null);
  const [data, setData] = useState<PolicyTelemetryTrends | null>(null);
  const [policyType, setPolicyType] = useState<PolicyVizType | "">("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTrends = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const result = await invoke<PolicyTelemetryTrends>("load_policy_telemetry_trends", {
        projectRoot,
        pythonExecutable: pythonPath || null,
        policyType: policyType || null,
        limit: 200,
      });
      setData(result);
    } catch (err) {
      setError(String(err));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, policyType]);

  useEffect(() => {
    void loadTrends();
  }, [loadTrends, refreshKey]);

  const rows = data?.rows ?? [];
  const compareOption = useMemo(
    () => buildTrendComparisonOption(rows, theme, logScale),
    [rows, theme, logScale]
  );
  const stepsOption = useMemo(() => buildTrendStepsOption(rows.slice(0, 12), theme), [rows, theme]);

  if (!projectRoot) return null;

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="flex items-center gap-2 text-sm font-semibold text-gray-200">
            <Database size={14} className="text-accent-primary" />
            Policy Telemetry Trends
            <span className="text-xs font-normal text-canvas-muted">
              SQLite cross-run store (§A.3 Option C)
            </span>
          </p>
          {data?.db_path && (
            <p className="text-[10px] text-canvas-muted mt-0.5 truncate max-w-xl">{data.db_path}</p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="input text-xs py-1"
            value={policyType}
            onChange={(e) => setPolicyType((e.target.value || "") as PolicyVizType | "")}
          >
            <option value="">All algorithm families</option>
            {(data?.policy_types ?? []).map((t) => (
              <option key={t} value={t}>
                {policyVizTypeLabel(t)}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn-secondary text-xs py-1 px-2 inline-flex items-center gap-1"
            onClick={() => void loadTrends()}
            disabled={loading}
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {error && <p className="text-xs text-accent-danger">{error}</p>}

      {rows.length === 0 && !loading && !error ? (
        <p className="text-xs text-canvas-muted">
          No persisted telemetry yet. Run a simulation with ALNS, HGS, or another
          ``PolicyVizMixin`` solver — snapshots are written to ``assets/telemetry.db`` on each emit.
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {compareOption ? (
              <div className="relative rounded-lg border border-canvas-border bg-canvas-elevated/40 p-2">
                <div className="absolute right-2 top-2 z-10">
                  <ChartExportButtons
                    chartRef={compareRef}
                    filenameStem="policy-telemetry-trends"
                  />
                </div>
                <ReactECharts
                  ref={compareRef}
                  option={compareOption}
                  style={{ height: 240, width: "100%" }}
                  notMerge
                  lazyUpdate
                />
              </div>
            ) : null}
            {stepsOption ? (
              <div className="relative rounded-lg border border-canvas-border bg-canvas-elevated/40 p-2">
                <div className="absolute right-2 top-2 z-10">
                  <ChartExportButtons chartRef={stepsRef} filenameStem="policy-telemetry-steps" />
                </div>
                <ReactECharts
                  ref={stepsRef}
                  option={stepsOption}
                  style={{ height: 240, width: "100%" }}
                  notMerge
                  lazyUpdate
                />
              </div>
            ) : null}
          </div>

          <div className="overflow-x-auto rounded-lg border border-canvas-border">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-canvas-elevated/60 text-canvas-muted">
                <tr>
                  <th className="px-2 py-1.5 font-medium">Run</th>
                  <th className="px-2 py-1.5 font-medium">Policy</th>
                  <th className="px-2 py-1.5 font-medium">Type</th>
                  <th className="px-2 py-1.5 font-medium">Day</th>
                  <th className="px-2 py-1.5 font-medium">Steps</th>
                  <th className="px-2 py-1.5 font-medium">Final metric</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 20).map((row) => (
                  <tr key={row.id} className="border-t border-canvas-border/60">
                    <td className="px-2 py-1.5 text-canvas-muted">{row.run_label ?? "—"}</td>
                    <td className="px-2 py-1.5 truncate max-w-[12rem]" title={row.policy}>
                      {row.policy}
                    </td>
                    <td className="px-2 py-1.5">{policyVizTypeLabel(row.policy_type)}</td>
                    <td className="px-2 py-1.5">{row.day}</td>
                    <td className="px-2 py-1.5">{row.step_count}</td>
                    <td className="px-2 py-1.5">{formatTrendMetric(row)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
