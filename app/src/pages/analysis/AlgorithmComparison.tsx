/**
 * Algorithm Comparison — side-by-side policy metric comparison.
 * Ports Streamlit `algorithms` mode.
 */
import ReactECharts from "echarts-for-react";
import { useMemo } from "react";
import { useGlobalFiltersStore } from "../../store/filters";
import { useSimStore, filterEntries } from "../../store/sim";

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
  const { entries } = useSimStore();
  const { policy, sampleId } = useGlobalFiltersStore();
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

  if (entries.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-canvas-muted text-sm">
        Load a simulation log in the Simulation Monitor to compare algorithms here.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="card">
        <p className="text-xs text-canvas-muted mb-3">Radar — normalised average metrics per policy</p>
        <ReactECharts option={radarOption} style={{ height: 340 }} />
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
