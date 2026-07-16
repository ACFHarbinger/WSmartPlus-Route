/**
 * Cross-run policy telemetry trends from ``assets/telemetry.db`` (§A.3 Option C).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Database, Download, RefreshCw } from "lucide-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { useAppStore } from "../../../store/app";
import { useGlobalFiltersStore } from "../../../store/filters";
import type {
  PolicyTelemetryTrends,
  PolicyTrajectoryTrends,
  PolicyVizType,
} from "../../../types";
import { isHighlighted } from "../../../utils/charts/chartHighlight";

import {
  buildTrendComparisonOption,
  buildTrendStepsOption,
  buildTrendTrajectoryOption,
  exportPolicyTelemetryTrendsCsv,
  exportPolicyTrajectoryCsv,
  filterTrajectorySeries,
  filterTrendRows,
  formatTrendMetric,
  policyVizTypeLabel,
  trendRowLabel,
  trendRowRunKey,
  trajectoryRunKey,
} from "../../../utils/benchmark/policyTelemetryTrends";

interface Props {
  theme: "dark" | "light";
  logScale?: boolean;
  /** Bump to reload trends after live telemetry emission. */
  refreshKey?: number;
  /** Pre-select trajectory policy filter (e.g. Simulation Monitor selection). */
  initialPolicy?: string | null;
  /** Pre-apply global run_label brush (e.g. portfolio single-run selection). */
  initialRunLabel?: string | null;
}

export function PolicyTelemetryTrendsPanel({
  theme,
  logScale = false,
  refreshKey = 0,
  initialPolicy = null,
  initialRunLabel = null,
}: Props) {
  const { projectRoot, pythonPath } = useAppStore();
  const globalPolicy = useGlobalFiltersStore((s) => s.policy);
  const globalRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const setPolicy = useGlobalFiltersStore((s) => s.setPolicy);
  const setRunLabel = useGlobalFiltersStore((s) => s.setRunLabel);
  const compareRef = useRef<EChartsReact | null>(null);
  const stepsRef = useRef<EChartsReact | null>(null);
  const trajectoryRef = useRef<EChartsReact | null>(null);
  const [data, setData] = useState<PolicyTelemetryTrends | null>(null);
  const [trajectories, setTrajectories] = useState<PolicyTrajectoryTrends | null>(null);
  const [policyType, setPolicyType] = useState<PolicyVizType | "">("");
  const [selectedPolicy, setSelectedPolicy] = useState("");
  const [smoothTrajectories, setSmoothTrajectories] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTrends = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const [result, traj] = await Promise.all([
        invoke<PolicyTelemetryTrends>("load_policy_telemetry_trends", {
          projectRoot,
          pythonExecutable: pythonPath || null,
          policyType: policyType || null,
          runLabel: globalRunLabel || null,
          limit: 200,
        }),
        invoke<PolicyTrajectoryTrends>("load_policy_trajectory_trends", {
          projectRoot,
          pythonExecutable: pythonPath || null,
          policy: selectedPolicy || null,
          policyType: policyType || null,
          runLabel: globalRunLabel || null,
          limit: 12,
        }),
      ]);
      setData(result);
      setTrajectories(traj);
    } catch (err) {
      setError(String(err));
      setData(null);
      setTrajectories(null);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, policyType, selectedPolicy, globalRunLabel]);

  useEffect(() => {
    void loadTrends();
  }, [loadTrends, refreshKey]);

  useEffect(() => {
    if (initialPolicy && !selectedPolicy) {
      setSelectedPolicy(initialPolicy);
    }
  }, [initialPolicy, selectedPolicy]);

  useEffect(() => {
    if (initialRunLabel && !globalRunLabel) {
      setRunLabel(initialRunLabel);
    }
  }, [initialRunLabel, globalRunLabel, setRunLabel]);

  const rows = data?.rows ?? [];
  const displayStepRows = useMemo(() => rows.slice(0, 12), [rows]);
  const allSeries = trajectories?.series ?? [];
  const filteredRows = useMemo(
    () => filterTrendRows(rows, globalPolicy, globalRunLabel),
    [rows, globalPolicy, globalRunLabel]
  );
  const filteredSeries = useMemo(
    () => filterTrajectorySeries(allSeries, globalPolicy, globalRunLabel),
    [allSeries, globalPolicy, globalRunLabel]
  );
  const brushedPolicies = useMemo(
    () => (globalPolicy ? [globalPolicy] : null),
    [globalPolicy]
  );
  const brushFilter = useMemo(
    () => ({ policy: globalPolicy, runLabel: globalRunLabel }),
    [globalPolicy, globalRunLabel]
  );

  const handlePolicyBrush = useCallback(
    (name: string) => {
      setPolicy(globalPolicy === name ? null : name);
    },
    [globalPolicy, setPolicy]
  );

  const handleRunBrush = useCallback(
    (label: string) => {
      setRunLabel(globalRunLabel === label ? null : label);
    },
    [globalRunLabel, setRunLabel]
  );
  const policyNames = useMemo(
    () => [...new Set(rows.map((r) => r.policy))].sort(),
    [rows]
  );
  const compareOption = useMemo(
    () => buildTrendComparisonOption(rows, theme, logScale, brushFilter),
    [rows, theme, logScale, brushFilter]
  );
  const stepsOption = useMemo(
    () => buildTrendStepsOption(displayStepRows, theme, brushFilter),
    [displayStepRows, theme, brushFilter]
  );
  const trajectoryOption = useMemo(
    () =>
      buildTrendTrajectoryOption(allSeries, theme, logScale, smoothTrajectories, brushFilter),
    [allSeries, theme, logScale, smoothTrajectories, brushFilter]
  );

  const handleComparisonClick = useCallback(
    (params: { seriesName?: string; name?: string }) => {
      if (params.name) handlePolicyBrush(params.name);
      if (params.seriesName) {
        const match = rows.find((row) => trendRowLabel(row) === params.seriesName);
        if (match) handleRunBrush(trendRowRunKey(match));
      }
    },
    [rows, handlePolicyBrush, handleRunBrush]
  );

  const handleStepsClick = useCallback(
    (params: { dataIndex?: number }) => {
      const row = params.dataIndex != null ? displayStepRows[params.dataIndex] : undefined;
      if (!row) return;
      handlePolicyBrush(row.policy);
      handleRunBrush(trendRowRunKey(row));
    },
    [displayStepRows, handlePolicyBrush, handleRunBrush]
  );

  const handleTrajectoryClick = useCallback(
    (params: { seriesIndex?: number }) => {
      const item = params.seriesIndex != null ? allSeries[params.seriesIndex] : undefined;
      if (!item) return;
      handlePolicyBrush(item.policy);
      handleRunBrush(trajectoryRunKey(item));
    },
    [allSeries, handlePolicyBrush, handleRunBrush]
  );

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
            <OpenPathToolbar
              path={data.db_path}
              projectRoot={projectRoot}
              chipClassName="mt-0.5 max-w-xl"
              handoff={false}
            />
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
          <select
            className="input text-xs py-1 max-w-[14rem]"
            value={selectedPolicy}
            onChange={(e) => setSelectedPolicy(e.target.value)}
          >
            <option value="">All policies (trajectories)</option>
            {policyNames.map((name) => (
              <option key={name} value={name}>
                {name.length > 36 ? `${name.slice(0, 34)}…` : name}
              </option>
            ))}
          </select>
          <label className="inline-flex items-center gap-1 text-[11px] text-canvas-muted">
            <input
              type="checkbox"
              className="rounded border-canvas-border"
              checked={smoothTrajectories}
              onChange={(e) => setSmoothTrajectories(e.target.checked)}
            />
            EMA smooth
          </label>
          <button
            type="button"
            className="btn-secondary text-xs py-1 px-2 inline-flex items-center gap-1"
            onClick={() => void loadTrends()}
            disabled={loading}
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          {rows.length > 0 && (
            <button
              type="button"
              className="btn-secondary text-xs py-1 px-2 inline-flex items-center gap-1"
              onClick={() => exportPolicyTelemetryTrendsCsv(filteredRows)}
            >
              <Download size={12} />
              Export CSV
            </button>
          )}
          {filteredSeries.length > 0 && (
            <button
              type="button"
              className="btn-secondary text-xs py-1 px-2 inline-flex items-center gap-1"
              onClick={() => exportPolicyTrajectoryCsv(filteredSeries)}
            >
              <Download size={12} />
              Trajectory CSV
            </button>
          )}
        </div>
      </div>

      {(globalPolicy || globalRunLabel) && (
        <p className="text-[11px] text-canvas-muted">
          Global brush active
          {globalPolicy ? ` · policy: ${globalPolicy}` : ""}
          {globalRunLabel ? ` · run: ${globalRunLabel}` : ""}
          {" — "}
          <button
            type="button"
            className="text-accent-primary hover:underline"
            onClick={() => {
              setPolicy(null);
              setRunLabel(null);
            }}
          >
            Clear brush
          </button>
        </p>
      )}

      {error && <p className="text-xs text-accent-danger">{error}</p>}

      {rows.length === 0 && !loading && !error ? (
        <p className="text-xs text-canvas-muted">
          No persisted telemetry yet. Run a simulation with ALNS, HGS, or another
          ``PolicyVizMixin`` solver — snapshots are written to ``assets/telemetry.db`` on each emit.
        </p>
      ) : filteredRows.length === 0 && (globalPolicy || globalRunLabel) ? (
        <p className="text-xs text-canvas-muted">
          No telemetry rows match the active global brush. Clear the brush or select a different
          policy / run.
        </p>
      ) : (
        <>
          {trajectoryOption ? (
            <div className="relative rounded-lg border border-canvas-border bg-canvas-elevated/40 p-2">
              <div className="absolute right-2 top-2 z-10">
                <ChartExportButtons
                  chartRef={trajectoryRef}
                  filenameStem="policy-telemetry-trajectories"
                />
              </div>
              <ReactECharts
                ref={trajectoryRef}
                option={trajectoryOption}
                style={{ height: 260, width: "100%" }}
                notMerge
                lazyUpdate
                onEvents={{ click: handleTrajectoryClick }}
              />
            </div>
          ) : null}

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
                  onEvents={{ click: handleComparisonClick }}
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
                  onEvents={{ click: handleStepsClick }}
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
                {filteredRows.slice(0, 20).map((row) => {
                  const runKey = trendRowRunKey(row);
                  const dimmed =
                    !isHighlighted(row.policy, brushedPolicies) ||
                    (globalRunLabel != null && runKey !== globalRunLabel);
                  return (
                    <tr
                      key={row.id}
                      className={`border-t border-canvas-border/60 transition-opacity ${
                        dimmed ? "opacity-35" : "opacity-100"
                      }`}
                    >
                      <td
                        className="px-2 py-1.5 text-canvas-muted cursor-pointer hover:text-accent-primary"
                        onClick={() => handleRunBrush(runKey)}
                        title="Click to brush run"
                      >
                        {row.run_label ?? "—"}
                      </td>
                      <td
                        className="px-2 py-1.5 truncate max-w-[12rem] cursor-pointer hover:text-accent-primary"
                        title={`${row.policy} — click to brush policy`}
                        onClick={() => handlePolicyBrush(row.policy)}
                      >
                        {row.policy}
                      </td>
                      <td className="px-2 py-1.5">{policyVizTypeLabel(row.policy_type)}</td>
                      <td className="px-2 py-1.5">{row.day}</td>
                      <td className="px-2 py-1.5">{row.step_count}</td>
                      <td className="px-2 py-1.5">{formatTrendMetric(row)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
