/**
 * Algorithm Comparison — side-by-side policy metric comparison.
 * Ports Streamlit `algorithms` mode.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { Download, Map } from "lucide-react";
import { SqlQueryPanel } from "../../components/analysis/SqlQueryPanel";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useAppStore } from "../../store/app";
import { useDuckDbStore } from "../../store/duckdb";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, filterEntries } from "../../store/sim";
import { formatPipelineTimingBadge, runSimulationArrowPipeline } from "../../utils/arrowPipeline";
import { barOpacity } from "../../utils/chartHighlight";
import { exportChartPng } from "../../utils/chartExport";
import { radarAxisValue } from "../../utils/chartLogScale";
import { symlog } from "../../utils/symlog";

const ALGORITHM_SIM_TABLE = "algorithm_sim";

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

const METRICS = [
  { key: "profit", label: "Profit (€)" },
  { key: "km", label: "Distance (km)" },
  { key: "overflows", label: "Overflows" },
  { key: "kg/km", label: "Efficiency (kg/km)" },
];

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

export function AlgorithmComparison() {
  const { entries, watchPath } = useSimStore();
  const { setMode, setPendingMapCompare, theme } = useAppStore();
  const { policy, sampleId, setPolicy } = useGlobalFiltersStore();
  const brushedPolicies = useMemo(() => (policy ? [policy] : null), [policy]);
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const radarRef = useRef<ReactECharts>(null);
  const barRefs = useRef<Record<string, EChartsReact | null>>({});
  const logScale = useGlobalFiltersStore((s) => s.logScale);

  useEffect(() => {
    if (!duckdbReady || !watchPath) return;
    setDuckdbLoading(true);
    runSimulationArrowPipeline(watchPath, ALGORITHM_SIM_TABLE)
      .then(setLastPipeline)
      .catch((err) => console.warn("Algorithm Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [watchPath, duckdbReady, setLastPipeline, setDuckdbLoading]);

  const filtered = useMemo(
    () => filterEntries(entries, policy, sampleId),
    [entries, policy, sampleId]
  );
  const policies = useMemo(() => [...new Set(filtered.map((e) => e.policy))], [filtered]);

  const radarOption = useMemo(() => {
    const metricMeans: Record<string, Record<string, number>> = {};
    for (const p of policies) {
      metricMeans[p] = {};
      for (const { key } of METRICS) {
        const vals = filtered
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[key] ?? 0);
        metricMeans[p][key] = mean(vals);
      }
    }

    const displayValue = (key: string, raw: number) => radarAxisValue(raw, key, logScale);
    const maxes = METRICS.map(({ key }) =>
      Math.max(...policies.map((p) => displayValue(key, metricMeans[p][key] ?? 0)), 1)
    );

    return {
      backgroundColor: "transparent",
      legend: { data: policies, textStyle: { color: "#9090b0" } },
      radar: {
        indicator: METRICS.map(({ label }, i) => ({ name: label, max: maxes[i] * 1.1 })),
        axisLine: { lineStyle: { color: "#2d2d50" } },
        splitLine: { lineStyle: { color: "#2d2d50" } },
        name: { textStyle: { color: "#9090b0" } },
      },
      series: [
        {
          type: "radar",
          data: policies.map((p, i) => ({
            name: p,
            value: METRICS.map(({ key }) => displayValue(key, metricMeans[p][key] ?? 0)),
            lineStyle: {
              color: COLORS[i % COLORS.length],
              opacity: barOpacity(p, brushedPolicies),
            },
            areaStyle: { color: `${COLORS[i % COLORS.length]}20` },
          })),
        },
      ],
      tooltip: {},
    };
  }, [filtered, policies, brushedPolicies, logScale]);

  const openOnMap = useCallback(() => {
    setPendingMapCompare({
      policies,
      layout: policies.length === 2 ? "split" : "overlay",
      mapMode: "deckgl",
    });
    setMode("simulation");
  }, [policies, setMode, setPendingMapCompare]);

  const handlePolicyClick = useCallback(
    (name: string) => {
      setPolicy(policy === name ? null : name);
    },
    [policy, setPolicy]
  );

  const onChartClick = useCallback(
    (params: { name?: string; seriesName?: string }) => {
      const name = params.name ?? params.seriesName;
      if (name) handlePolicyClick(name);
    },
    [handlePolicyClick]
  );

  if (entries.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Load a simulation log in the Simulation Monitor to compare algorithms here.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale />

      <div className="flex items-center gap-3 flex-wrap">
        {watchPath && (
          <span className="text-xs text-canvas-muted font-mono truncate">
            {watchPath.split("/").pop()}
            {!duckdbLoading && lastPipeline?.tableName === ALGORITHM_SIM_TABLE && (
              <> · {formatPipelineTimingBadge(lastPipeline)}</>
            )}
          </span>
        )}
        <button onClick={openOnMap} className="btn-ghost text-xs flex items-center gap-1.5">
          <Map size={12} />
          Compare on Map
        </button>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-canvas-muted">
            Radar — {logScale ? "log-normalised" : "normalised"} average metrics per policy
          </p>
          <button
            onClick={() => exportChartPng(radarRef, "algorithm-radar.png")}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Download size={12} />
            PNG
          </button>
        </div>
        <ReactECharts
          ref={radarRef}
          option={radarOption}
          style={{ height: 340 }}
          onEvents={{ click: onChartClick, legendselectchanged: onChartClick }}
        />
      </div>

      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Log-scale bars — profit · km · symlog-overflows · kg/km per policy"
          : "Linear bars — profit · km · overflows · kg/km per policy"}
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {METRICS.map(({ key, label }) => {
          const symlogOverflows = logScale && key === "overflows";
          const option = {
            backgroundColor: "transparent",
            grid: { left: 50, right: 10, top: 10, bottom: 40 },
            xAxis: {
              type: "category",
              data: policies,
              axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
            },
            yAxis: {
              type: (logScale && !symlogOverflows ? "log" : "value") as "log" | "value",
              logBase: 10,
              axisLabel: { color: "#9090b0", fontSize: 10 },
              minorSplitLine: { show: false },
            },
            series: [
              {
                type: "bar",
                data: policies.map((p, i) => {
                  const raw = mean(
                    filtered
                      .filter((e) => e.policy === p)
                      .map((e) => (e.data as Record<string, number>)[key] ?? 0)
                  );
                  const value = !logScale
                    ? raw
                    : symlogOverflows
                      ? symlog(raw)
                      : Math.max(raw, 0.001);
                  return {
                    value,
                    itemStyle: {
                      color: COLORS[i % COLORS.length],
                      opacity: barOpacity(p, brushedPolicies),
                    },
                  };
                }),
              },
            ],
            tooltip: { trigger: "axis" },
          };
          return (
            <div key={key} className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">{label}</p>
                <button
                  onClick={() => exportChartPng({ current: barRefs.current[key] }, `algorithm-${key}.png`)}
                  className="btn-ghost text-xs flex items-center gap-1"
                >
                  <Download size={12} />
                  PNG
                </button>
              </div>
              <ReactECharts
                ref={(el) => {
                  barRefs.current[key] = el;
                }}
                option={option}
                style={{ height: 140 }}
                onEvents={{ click: onChartClick }}
              />
            </div>
          );
        })}
      </div>

      {watchPath && duckdbReady && (
        <SqlQueryPanel
          tableName={ALGORITHM_SIM_TABLE}
          theme={theme}
          highlightPolicies={brushedPolicies}
          brushSqlSync
          autoRunOnBrushSync
          algorithmMode
        />
      )}
    </div>
  );
}
