/**
 * Algorithm Comparison — side-by-side policy metric comparison.
 * Ports Streamlit `algorithms` mode.
 */
import { useCallback, useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { Download, Map } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, filterEntries } from "../../store/sim";
import { exportChartPng } from "../../utils/chartExport";

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
  const { setMode } = useAppStore();
  const { policy, sampleId } = useGlobalFiltersStore();
  const radarRef = useRef<ReactECharts>(null);

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

    const maxes = METRICS.map(({ key }) =>
      Math.max(...policies.map((p) => metricMeans[p][key] ?? 0), 1)
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
            value: METRICS.map(({ key }) => metricMeans[p][key] ?? 0),
            lineStyle: { color: COLORS[i % COLORS.length] },
            areaStyle: { color: `${COLORS[i % COLORS.length]}20` },
          })),
        },
      ],
      tooltip: {},
    };
  }, [filtered, policies]);

  const openOnMap = useCallback(() => {
    setMode("simulation");
  }, [setMode]);

  if (entries.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Load a simulation log in the Simulation Monitor to compare algorithms here.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar />

      <div className="flex items-center gap-3 flex-wrap">
        {watchPath && (
          <span className="text-xs text-canvas-muted font-mono truncate">
            {watchPath.split("/").pop()}
          </span>
        )}
        <button onClick={openOnMap} className="btn-ghost text-xs flex items-center gap-1.5">
          <Map size={12} />
          Compare on Map
        </button>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-canvas-muted">Radar — normalised average metrics per policy</p>
          <button
            onClick={() => exportChartPng(radarRef, "algorithm-radar.png")}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Download size={12} />
            PNG
          </button>
        </div>
        <ReactECharts ref={radarRef} option={radarOption} style={{ height: 340 }} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {METRICS.map(({ key, label }) => {
          const option = {
            backgroundColor: "transparent",
            grid: { left: 50, right: 10, top: 10, bottom: 40 },
            xAxis: {
              type: "category",
              data: policies,
              axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
            },
            yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 10 } },
            series: [
              {
                type: "bar",
                data: policies.map((p, i) => ({
                  value: mean(
                    filtered
                      .filter((e) => e.policy === p)
                      .map((e) => (e.data as Record<string, number>)[key] ?? 0)
                  ),
                  itemStyle: { color: COLORS[i % COLORS.length] },
                })),
              },
            ],
            tooltip: { trigger: "axis" },
          };
          return (
            <div key={key} className="card">
              <p className="text-xs text-canvas-muted mb-2">{label}</p>
              <ReactECharts option={option} style={{ height: 140 }} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
