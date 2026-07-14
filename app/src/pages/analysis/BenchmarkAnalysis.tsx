/**
 * Benchmark Analysis — multi-run, multi-policy comparison.
 * Ports Streamlit `benchmark` mode.
 *
 * Also consumes `pendingEvalResults` from EvaluationRunner (§G.12) to display
 * checkpoint comparison charts without requiring simulation log files.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, X, Download } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { exportChartPng } from "../../utils/chartExport";
import { paretoFront, paretoStepLine } from "../../utils/pareto";
import { PARETO_PANELS, panelForRun } from "../../utils/paretoPanels";
import { cityScaleLabel, parseLogPath, parsePolicyLabel, strategyColor } from "../../utils/simMetadata";
import { symlog } from "../../utils/symlog";
import { scanOutputPortfolio } from "../../utils/outputRunLogs";
import { downloadCsv } from "../../utils/tableExport";
import { toast } from "sonner";
import type { DayLogEntry, EvalAnalyticsRow } from "../../types";

interface RunFile {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function fmt(n: number, d = 1) {
  return isFinite(n) ? n.toFixed(d) : "—";
}

function BenchmarkParetoPanel({
  label,
  points,
  logScale,
}: {
  label: string;
  points: Array<{ id: string; x: number; y: number; policy: string }>;
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const option = useMemo(() => {
    const frontIds = new Set(paretoFront(points).map((p) => p.id));
    const step = paretoStepLine(paretoFront(points));
    const displayY = (y: number) => (logScale ? Math.max(y, 0.001) : y);

    return {
      backgroundColor: "transparent",
      title: { text: label, left: "center", textStyle: { color: "#9090b0", fontSize: 10 } },
      grid: { left: 44, right: 10, top: 36, bottom: 32 },
      xAxis: {
        type: "value",
        name: "Profit",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      },
      yAxis: {
        type: logScale ? "log" : "value",
        logBase: 10,
        name: "Ovfl",
        nameTextStyle: { color: "#9090b0", fontSize: 8 },
        axisLabel: { color: "#9090b0", fontSize: 8 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "scatter",
          data: points.map((pt) => ({
            name: pt.policy,
            value: [pt.x, displayY(pt.y)],
            itemStyle: {
              color: strategyColor(pt.policy, { [pt.policy]: parsePolicyLabel(pt.policy) }),
            },
            symbolSize: frontIds.has(pt.id) ? 9 : 6,
          })),
          tooltip: {
            formatter: (p: { name: string; value: [number, number] }) =>
              `${p.name}<br/>Profit: ${fmt(p.value[0])} €<br/>Overflows: ${fmt(points.find((x) => x.policy === p.name)?.y ?? p.value[1])}`,
          },
        },
        ...(step.length > 1
          ? [
              {
                type: "line",
                data: step.map(([x, y]) => [x, displayY(y)]),
                lineStyle: { color: "#f3f4f6", type: "dashed", width: 1 },
                symbol: "none",
                tooltip: { show: false },
                z: 1,
              },
            ]
          : []),
      ],
    };
  }, [label, points, logScale]);

  if (points.length === 0) {
    return (
      <div className="card flex items-center justify-center h-[200px] text-xs text-canvas-muted">
        {label} — no runs
      </div>
    );
  }

  return (
    <div className="card">
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}

const GRAPH_HEAT_METRICS = [
  { key: "profit", label: "Profit", higherBetter: true },
  { key: "kg/km", label: "kg/km", higherBetter: true },
  { key: "overflows", label: "Overflows", higherBetter: false },
] as const;

function BenchmarkPortfolioParallel({ runs }: { runs: RunFile[] }) {
  const lines = useMemo(
    () =>
      runs.map((run, i) => {
        const meta = parseLogPath(run.path);
        const vals = run.entries.map((e) => e.data as Record<string, number>);
        const profit = mean(vals.map((d) => d.profit).filter((v): v is number => v != null));
        const kgkm = mean(vals.map((d) => d["kg/km"]).filter((v): v is number => v != null));
        const overflows = mean(vals.map((d) => d.overflows).filter((v): v is number => v != null));
        const km = mean(vals.map((d) => d.km).filter((v): v is number => v != null));
        return {
          name: run.label,
          color: COLORS[i % COLORS.length],
          value: [meta.scale ?? 100, profit, kgkm, overflows, km],
          meta,
        };
      }),
    [runs]
  );

  const option = useMemo(() => {
    const schema = [
      { dim: 0, name: "N", max: Math.max(...lines.map((l) => l.value[0]), 1) * 1.1 },
      { dim: 1, name: "Profit", max: Math.max(...lines.map((l) => l.value[1]), 1) * 1.1 },
      { dim: 2, name: "kg/km", max: Math.max(...lines.map((l) => l.value[2]), 0.01) * 1.1 },
      { dim: 3, name: "Overflows", max: Math.max(...lines.map((l) => l.value[3]), 1) * 1.1 },
      { dim: 4, name: "km", max: Math.max(...lines.map((l) => l.value[4]), 1) * 1.1 },
    ];

    return {
      backgroundColor: "transparent",
      parallelAxis: schema.map((s) => ({
        dim: s.dim,
        name: s.name,
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLine: { lineStyle: { color: "#2d2d50" } },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      })),
      parallel: { left: 50, right: 24, top: 28, bottom: 36 },
      series: [
        {
          type: "parallel",
          lineStyle: { width: 1.5, opacity: 0.75 },
          data: lines.map((l) => ({
            name: l.name,
            value: l.value,
            lineStyle: { color: l.color },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const line = lines.find((l) => l.name === p.name);
          if (!line) return p.name;
          return [
            p.name,
            cityScaleLabel(line.meta),
            line.meta.distribution,
            line.meta.improver,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [lines]);

  return (
    <div className="card space-y-2">
      <p className="text-xs font-semibold text-gray-300">Portfolio Parallel Coordinates (§G.1.4)</p>
      <p className="text-[10px] text-canvas-muted">
        {runs.length} simulation log(s) — one polyline per loaded run
      </p>
      <ReactECharts option={option} style={{ height: Math.min(360, 120 + runs.length * 4) }} />
    </div>
  );
}

function BenchmarkGraphHeatmap({
  graphLabel,
  runs,
  heatmapMode,
}: {
  graphLabel: string;
  runs: RunFile[];
  heatmapMode: "overflows" | "kg/km";
}) {
  const policies = useMemo(
    () => [...new Set(runs.flatMap((r) => r.entries.map((e) => e.policy)))],
    [runs]
  );

  const option = useMemo(() => {
    const metricKey = heatmapMode;
    const metric = GRAPH_HEAT_METRICS.find((m) => m.key === metricKey) ?? GRAPH_HEAT_METRICS[0];
    const raw = policies.map((p) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[metricKey])
          .filter((v): v is number => v != null)
      );
      return mean(vals);
    });
    const min = Math.min(...raw);
    const max = Math.max(...raw);
    const span = max - min || 1;
    const cells: Array<[number, number, number]> = raw.map((v, pi) => {
      let norm = (v - min) / span;
      if (!metric.higherBetter) norm = 1 - norm;
      return [pi, 0, norm];
    });

    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 16, top: 8, bottom: 40 },
      xAxis: {
        type: "category",
        data: policies,
        axisLabel: { color: "#9090b0", fontSize: 8, rotate: 25 },
      },
      yAxis: {
        type: "category",
        data: [metric.label],
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
        textStyle: { color: "#9090b0", fontSize: 9 },
        show: false,
      },
      series: [{ type: "heatmap", data: cells, label: { show: false } }],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const pi = p.value[0];
          return `${policies[pi]}<br/>${metric.label}: ${fmt(raw[pi], 2)}`;
        },
      },
    };
  }, [policies, runs, heatmapMode]);

  return (
    <div className="card space-y-1">
      <p className="text-[10px] text-canvas-muted font-mono">{graphLabel}</p>
      <ReactECharts option={option} style={{ height: 100 }} />
    </div>
  );
}

const SIM_METRICS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
  { key: "kg/km", label: "Efficiency (kg/km)", lowerIsBetter: false },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
];

const EVAL_METRICS = [
  { key: "cost", label: "Tour Cost", lowerIsBetter: true },
  { key: "gap", label: "Optimality Gap (%)", lowerIsBetter: true },
  { key: "time", label: "Time (s)", lowerIsBetter: true },
];

const COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f87171", "#818cf8", "#a3e635"];

function EvalResultsPanel({ rows, onDismiss }: { rows: EvalAnalyticsRow[]; onDismiss: () => void }) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const checkpoints = rows.map((r) => r.checkpoint);

  const makeBarOption = (metricKey: string, metricLabel: string) => ({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 55 },
    xAxis: {
      type: "category",
      data: checkpoints,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: "value",
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: rows.map((r) => (r[metricKey] as number | undefined) ?? 0),
        itemStyle: {
          color: (params: { dataIndex: number }) => COLORS[params.dataIndex % COLORS.length],
        },
      },
    ],
    tooltip: { trigger: "axis" },
  });

  return (
    <div className="space-y-4">
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-200">Eval Results — Checkpoint Comparison</p>
          <p className="text-xs text-canvas-muted mt-0.5">
            {rows.length} checkpoint(s) loaded from Evaluation Runner
          </p>
        </div>
        <button onClick={onDismiss} className="btn-ghost text-xs flex items-center gap-1">
          <X size={12} />
          Dismiss
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {EVAL_METRICS.map(({ key, label }) => (
          <div key={key} className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-canvas-muted">{label}</p>
              <button
                onClick={() => exportChartPng({ current: chartRefs.current[key] }, `eval-${key}.png`)}
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <Download size={12} />
                PNG
              </button>
            </div>
            <ReactECharts
              ref={(el) => {
                chartRefs.current[key] = el;
              }}
              option={makeBarOption(key, label)}
              style={{ height: 220 }}
            />
          </div>
        ))}
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Checkpoint</th>
              {EVAL_METRICS.map(({ key, label }) => (
                <th key={key} className="text-right py-2 px-3 text-canvas-muted font-medium">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border/30">
            {rows.map((r) => (
              <tr key={r.checkpoint} className="hover:bg-canvas-hover/40">
                <td className="py-1.5 px-3 font-mono text-gray-300">{r.checkpoint}</td>
                {EVAL_METRICS.map(({ key }) => (
                  <td key={key} className="py-1.5 px-3 text-right font-mono text-gray-400">
                    {typeof r[key] === "number" ? (r[key] as number).toFixed(4) : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function BenchmarkAnalysis() {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const {
    pendingEvalResults, setPendingEvalResults,
    pendingBenchmarkLogs, setPendingBenchmarkLogs,
    projectRoot,
  } = useAppStore();
  const [runs, setRuns] = useState<RunFile[]>([]);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [evalRows, setEvalRows] = useState<EvalAnalyticsRow[] | null>(null);
  const [logScale, setLogScale] = useState(false);
  const [graphHeatmapMode, setGraphHeatmapMode] = useState<"overflows" | "kg/km">("overflows");
  const { policy, sampleId } = useGlobalFiltersStore();

  const filteredRuns = useMemo(
    () =>
      runs.map((r) => ({
        ...r,
        entries: filterEntries(r.entries, policy, sampleId),
      })),
    [runs, policy, sampleId]
  );

  // Consume pending eval results on mount
  useEffect(() => {
    if (pendingEvalResults && pendingEvalResults.length > 0) {
      setEvalRows(pendingEvalResults);
      setPendingEvalResults(null);
    }
  }, [pendingEvalResults, setPendingEvalResults]);

  // Consume pending benchmark logs from Output Browser compare action
  useEffect(() => {
    if (!pendingBenchmarkLogs || pendingBenchmarkLogs.length === 0) return;
    const load = async () => {
      const loaded: RunFile[] = [];
      for (const ref of pendingBenchmarkLogs) {
        const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path: ref.path });
        loaded.push({ path: ref.path, label: ref.label, entries });
      }
      setRuns(loaded);
      setPendingBenchmarkLogs(null);
    };
    load().catch(console.error);
  }, [pendingBenchmarkLogs, setPendingBenchmarkLogs]);

  const addRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    const label = path.split("/").slice(-2).join("/");
    setRuns((r) => [...r, { path, label, entries }]);
  }, []);

  const removeRun = (path: string) => setRuns((r) => r.filter((x) => x.path !== path));

  const loadOutputPortfolio = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to scan output portfolio");
      return;
    }
    setPortfolioLoading(true);
    try {
      const refs = await scanOutputPortfolio(`${projectRoot}/assets/output`, 48);
      if (!refs.length) {
        toast.error("No simulation logs found under assets/output");
        return;
      }
      const loaded: RunFile[] = [];
      for (const ref of refs) {
        const entries = await invoke<DayLogEntry[]>("load_simulation_log", { path: ref.path });
        loaded.push({ path: ref.path, label: ref.label, entries });
      }
      setRuns(loaded);
      toast.success(`Loaded ${loaded.length} simulation log(s) from output portfolio`);
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot]);

  const exportComparisonCsv = useCallback(() => {
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];
    downloadCsv(
      "benchmark-comparison.csv",
      ["run", "policy", ...SIM_METRICS.map((m) => m.key)],
      filteredRuns.flatMap((r) =>
        policies.map((p) => {
          const vals = r.entries
            .filter((e) => e.policy === p)
            .map((e) => e.data as Record<string, number>);
          const row: Array<string | number> = [r.label, p];
          for (const { key } of SIM_METRICS) {
            const v = vals.map((d) => d[key]).filter((x): x is number => x != null);
            row.push(v.length ? mean(v).toFixed(4) : "");
          }
          return row;
        })
      )
    );
  }, [filteredRuns]);

  const efficiencyRanking = useMemo(() => {
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];
    return policies
      .map((p) => {
        const vals = filteredRuns.flatMap((r) =>
          r.entries
            .filter((e) => e.policy === p)
            .map((e) => (e.data as Record<string, number>)["kg/km"])
            .filter((v): v is number => v != null)
        );
        return { policy: p, value: mean(vals) };
      })
      .filter((r) => r.value > 0)
      .sort((a, b) => b.value - a.value);
  }, [filteredRuns]);

  const paretoByPanel = useMemo(() => {
    const panels: Record<string, Array<{ id: string; x: number; y: number; policy: string }>> = {};
    for (const panel of PARETO_PANELS) panels[panel.id] = [];

    for (const run of filteredRuns) {
      const panelId = panelForRun(parseLogPath(run.path));
      if (!panelId) continue;
      const policies = [...new Set(run.entries.map((e) => e.policy))];
      for (const p of policies) {
        const vals = run.entries.filter((e) => e.policy === p);
        const profit = mean(vals.map((e) => e.data.profit).filter((v): v is number => v != null));
        const overflows = mean(
          vals.map((e) => e.data.overflows).filter((v): v is number => v != null)
        );
        panels[panelId].push({
          id: `${run.path}::${p}`,
          policy: p,
          x: profit,
          y: overflows,
        });
      }
    }
    return panels;
  }, [filteredRuns]);

  const cityGroups = useMemo(() => {
    const map = new Map<string, RunFile[]>();
    for (const run of filteredRuns) {
      const label = cityScaleLabel(parseLogPath(run.path));
      const list = map.get(label) ?? [];
      list.push(run);
      map.set(label, list);
    }
    return [...map.entries()];
  }, [filteredRuns]);

  const cityComparisonOption = useMemo(() => {
    const labels = cityGroups.map(([city]) => city);
    const profitSeries = cityGroups.map(([, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries.map((e) => e.data.profit).filter((v): v is number => v != null)
      );
      return Math.max(mean(vals), 0.001);
    });
    const overflowSeries = cityGroups.map(([, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries.map((e) => e.data.overflows).filter((v): v is number => v != null)
      );
      return symlog(mean(vals));
    });

    return {
      backgroundColor: "transparent",
      legend: { textStyle: { color: "#9090b0", fontSize: 10 } },
      grid: { left: 50, right: 10, top: 30, bottom: 40 },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: "log",
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          name: "Mean profit (€)",
          type: "bar",
          data: profitSeries,
          itemStyle: { color: "#6366f1" },
        },
        {
          name: "Mean overflows (symlog)",
          type: "bar",
          data: overflowSeries,
          itemStyle: { color: "#f87171" },
        },
      ],
      tooltip: { trigger: "axis" },
    };
  }, [cityGroups]);

  const efficiencyRankOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      grid: { left: 110, right: 24, top: 12, bottom: 12 },
      xAxis: {
        type: "value",
        name: "kg/km",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: "category",
        data: efficiencyRanking.map((r) => r.policy),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      series: [
        {
          type: "bar",
          data: efficiencyRanking.map((r, i) => ({
            value: r.value,
            itemStyle: { color: COLORS[i % COLORS.length] },
          })),
        },
      ],
      tooltip: { trigger: "axis" },
    }),
    [efficiencyRanking]
  );

  const makeBarOption = (metricKey: string, metricLabel: string) => {
    const runLabels = filteredRuns.map((r) => r.label);
    const policies = [...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy)))];

    const series = policies.map((p, i) => ({
      name: p,
      type: "bar",
      data: filteredRuns.map((r) => {
        const vals = r.entries
          .filter((e) => e.policy === p)
          .map((e) => (e.data as Record<string, number>)[metricKey] ?? null)
          .filter((v): v is number => v !== null);
        return mean(vals);
      }),
      itemStyle: { color: COLORS[i % COLORS.length] },
    }));

    return {
      backgroundColor: "transparent",
      legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
      grid: { left: 50, right: 10, top: 35, bottom: 55 },
      xAxis: {
        type: "category",
        data: runLabels,
        axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
      },
      yAxis: {
        type: logScale ? "log" : "value",
        logBase: 10,
        name: metricLabel,
        nameTextStyle: { color: "#9090b0" },
        axisLabel: { color: "#9090b0", fontSize: 10 },
        minorSplitLine: { show: false },
      },
      series: series.map((s) => ({
        ...s,
        data: (s.data as number[]).map((v) => (logScale ? Math.max(v, 0.001) : v)),
      })),
      tooltip: { trigger: "axis" },
    };
  };

  return (
    <div className="space-y-4">
      <GlobalFilterBar />

      {evalRows && evalRows.length > 0 && (
        <EvalResultsPanel rows={evalRows} onDismiss={() => setEvalRows(null)} />
      )}

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={addRun} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Add Simulation Run
        </button>
        {projectRoot && (
          <button
            onClick={() => void loadOutputPortfolio()}
            disabled={portfolioLoading}
            className="btn-ghost flex items-center gap-2 text-xs"
          >
            <FolderOpen size={14} />
            {portfolioLoading ? "Scanning output…" : "Load output portfolio (§G.1.4)"}
          </button>
        )}
        <span className="text-xs text-canvas-muted">{runs.length} simulation run(s) loaded</span>
        {filteredRuns.length > 0 && (
          <button onClick={exportComparisonCsv} className="btn-ghost text-xs flex items-center gap-1">
            <Download size={12} />
            Export CSV
          </button>
        )}
      </div>

      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Loaded Runs
          </p>
          <div className="space-y-1">
            {runs.map((r) => (
              <div key={r.path} className="flex items-center gap-2 text-xs text-gray-300">
                <button
                  onClick={() => removeRun(r.path)}
                  className="text-canvas-muted hover:text-accent-danger"
                >
                  <X size={12} />
                </button>
                <span className="font-mono truncate">{r.label}</span>
                <span className="ml-auto text-canvas-muted">{r.entries.length} days</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {runs.length === 0 && !evalRows && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Add simulation run logs to compare, or use &quot;Open in Analytics →&quot; from the Evaluation Runner.
        </div>
      )}

      {runs.length >= 1 && (
        <div className="flex justify-end">
          <button
            onClick={() => setLogScale((v) => !v)}
            className={`btn-ghost text-xs ${logScale ? "text-accent-secondary" : ""}`}
          >
            {logScale ? "Log scale (on)" : "Log scale (off)"}
          </button>
        </div>
      )}

      {runs.length >= 2 && (
        <BenchmarkPortfolioParallel runs={filteredRuns} />
      )}

      {runs.length >= 1 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-300">Pareto Panels (§G.1.2)</p>
          <div className="grid grid-cols-2 gap-4">
            {PARETO_PANELS.map((panel) => (
              <BenchmarkParetoPanel
                key={panel.id}
                label={panel.label}
                points={paretoByPanel[panel.id] ?? []}
                logScale={logScale}
              />
            ))}
          </div>
        </div>
      )}

      {runs.length >= 2 && cityGroups.length > 1 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-xs font-semibold text-gray-300">Heatmaps by Graph (§G.1.3)</p>
            <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
              {(["overflows", "kg/km"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setGraphHeatmapMode(m)}
                  className={`text-xs px-2.5 py-1 rounded-md ${
                    graphHeatmapMode === m
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {cityGroups.map(([graph, graphRuns]) => (
              <BenchmarkGraphHeatmap
                key={graph}
                graphLabel={graph}
                runs={graphRuns}
                heatmapMode={graphHeatmapMode}
              />
            ))}
          </div>
        </div>
      )}

      {runs.length >= 2 && cityGroups.length >= 1 && (
        <div className="card space-y-2">
          <p className="text-xs font-semibold text-gray-300">City Comparison (§G.1.6)</p>
          <p className="text-[10px] text-canvas-muted">Log scale only — preserves extreme values</p>
          <ReactECharts
            ref={(el) => {
              chartRefs.current["city-compare"] = el;
            }}
            option={cityComparisonOption}
            style={{ height: 240 }}
          />
        </div>
      )}

      {runs.length >= 1 && efficiencyRanking.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">Efficiency Ranking (kg/km)</p>
            <button
              onClick={() =>
                exportChartPng({ current: chartRefs.current["efficiency-rank"] }, "benchmark-efficiency-rank.png")
              }
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              PNG
            </button>
          </div>
          <ReactECharts
            ref={(el) => {
              chartRefs.current["efficiency-rank"] = el;
            }}
            option={efficiencyRankOption}
            style={{ height: Math.max(180, efficiencyRanking.length * 28) }}
          />
        </div>
      )}

      {runs.length >= 1 && (
        <div className="grid grid-cols-2 gap-4">
          {SIM_METRICS.map(({ key, label }) => (
            <div key={key} className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">{label}</p>
                <button
                  onClick={() => exportChartPng({ current: chartRefs.current[key] }, `benchmark-${key}.png`)}
                  className="btn-ghost text-xs flex items-center gap-1"
                >
                  <Download size={12} />
                  PNG
                </button>
              </div>
              <ReactECharts
                ref={(el) => {
                  chartRefs.current[key] = el;
                }}
                option={makeBarOption(key, label)}
                style={{ height: 220 }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
