/**
 * Training Monitor — live and historical training metrics (§G.17).
 *
 * Ports the Streamlit `training` mode from:
 *   logic/src/ui/pages/training.py
 *   logic/src/ui/pages/training_charts.py
 *   logic/src/ui/services/data_loader.py (discover_training_runs, load_training_metrics)
 *
 * §G.17 additions in this pass:
 *   - Multi-run overlay chart: all selected runs on one ECharts canvas, colour-coded
 *   - Hyperparameter panel: reads hparams.yaml via read_text_file, flat key-value table
 *   - Gradient norm sparkline: separate chart for the grad_norm column
 *   - Checkpoint browser: lists checkpoints/ dir, "Load in Eval Runner" button
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { ChevronDown, ChevronRight, FolderOpen, RefreshCw } from "lucide-react";
import { useAppStore } from "../store/app";
import type { DirEntry, TrainingRun, TrainingMetricsRow } from "../types";

// ── Colour palette for multi-run overlay
const RUN_COLORS = [
  "#6366f1", // indigo
  "#34d399", // emerald
  "#f87171", // red
  "#fbbf24", // amber
  "#a78bfa", // violet
  "#fb923c", // orange
  "#38bdf8", // sky
  "#f472b6", // pink
];

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// ── Multi-run overlay chart (train_loss + reward per run on shared axes)
function MultiRunChart({
  runsMetrics,
}: {
  runsMetrics: { name: string; metrics: TrainingMetricsRow[] }[];
}) {
  if (runsMetrics.length === 0) return null;

  const series: object[] = [];
  const legendData: string[] = [];

  runsMetrics.forEach(({ name, metrics }, i) => {
    const color = RUN_COLORS[i % RUN_COLORS.length];
    const epochs = metrics.map((r) => r.epoch ?? r.step ?? 0);
    const trainLoss = metrics.map((r, j) => [epochs[j], r.train_loss ?? null]);
    const valLoss = metrics.map((r, j) => [epochs[j], r.val_loss ?? null]);
    const reward = metrics.map((r, j) => [epochs[j], r.reward ?? null]);

    const hasReward = metrics.some((r) => r.reward != null);
    const shortName = name.length > 20 ? `…${name.slice(-18)}` : name;

    legendData.push(`${shortName} loss`, ...(hasReward ? [`${shortName} reward`] : []));

    series.push({
      name: `${shortName} loss`,
      type: "line",
      data: trainLoss,
      smooth: true,
      lineStyle: { color, width: 2 },
      itemStyle: { color },
      symbol: "none",
      encode: { x: 0, y: 1 },
    });

    if (metrics.some((r) => r.val_loss != null)) {
      series.push({
        name: `${shortName} val`,
        type: "line",
        data: valLoss,
        smooth: true,
        lineStyle: { color, width: 1.5, type: "dashed" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }

    if (hasReward) {
      series.push({
        name: `${shortName} reward`,
        type: "line",
        yAxisIndex: 1,
        data: reward,
        smooth: true,
        lineStyle: { color, width: 2, type: "dotted" },
        itemStyle: { color },
        symbol: "none",
        encode: { x: 0, y: 1 },
      });
    }
  });

  return (
    <div className="card">
      <ReactECharts
        option={{
          backgroundColor: "transparent",
          legend: {
            data: legendData,
            textStyle: { color: "#9090b0", fontSize: 10 },
            type: "scroll",
          },
          grid: { left: 50, right: 60, top: 45, bottom: 30 },
          xAxis: {
            type: "value",
            name: "Epoch",
            nameTextStyle: { color: "#9090b0" },
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: [
            {
              type: "value",
              name: "Loss",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
            },
            {
              type: "value",
              name: "Reward",
              nameTextStyle: { color: "#9090b0" },
              axisLabel: { color: "#9090b0", fontSize: 10 },
              splitLine: { show: false },
            },
          ],
          series,
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 260 }}
      />
    </div>
  );
}

// ── Gradient norm sparkline
function GradNormSparkline({ metrics }: { metrics: TrainingMetricsRow[] }) {
  const data = metrics
    .filter((r) => r.grad_norm != null)
    .map((r) => [r.epoch ?? r.step ?? 0, r.grad_norm!]);

  if (data.length === 0) return null;

  return (
    <div className="card">
      <p className="text-xs text-canvas-muted mb-1">Gradient Norm</p>
      <ReactECharts
        option={{
          backgroundColor: "transparent",
          grid: { left: 40, right: 10, top: 8, bottom: 28 },
          xAxis: {
            type: "value",
            axisLabel: { color: "#9090b0", fontSize: 9 },
          },
          yAxis: { type: "value", axisLabel: { color: "#9090b0", fontSize: 9 } },
          series: [{
            type: "line",
            data,
            smooth: false,
            lineStyle: { color: "#f87171", width: 1.5 },
            areaStyle: { color: "rgba(248,113,113,0.1)" },
            symbol: "none",
          }],
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 80 }}
      />
    </div>
  );
}

// ── Hyperparameter panel (reads hparams.yaml, renders flat key-value table)
function HparamsPanel({ runPath }: { runPath: string }) {
  const [pairs, setPairs] = useState<{ key: string; value: string }[] | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const hparamsPath = `${runPath}/hparams.yaml`;
    invoke<string>("read_text_file", { path: hparamsPath })
      .then((text) => {
        const parsed = text
          .split("\n")
          .map((line) => line.match(/^(\S[^:]*?):\s*(.*)$/))
          .filter(Boolean)
          .map((m) => ({ key: m![1].trim(), value: m![2].trim() }))
          .filter(({ key }) => !key.startsWith("#"));
        setPairs(parsed);
      })
      .catch(() => setPairs(null));
  }, [runPath]);

  if (!pairs?.length) return null;

  const visible = expanded ? pairs : pairs.slice(0, 8);

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-canvas-surface hover:bg-canvas-hover text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Hyperparameters
        </span>
        <span className="ml-auto text-xs text-canvas-muted">{pairs.length} params</span>
      </button>
      {expanded && (
        <div className="px-4 py-2 bg-canvas-bg border-t border-canvas-border">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
            {visible.map(({ key, value }) => (
              <>
                <dt key={`k-${key}`} className="text-xs text-canvas-muted font-mono py-0.5">{key}</dt>
                <dd key={`v-${key}`} className="text-xs text-gray-300 font-mono py-0.5 truncate">{value || "—"}</dd>
              </>
            ))}
          </dl>
          {pairs.length > 8 && (
            <button
              className="text-xs text-accent-primary mt-2"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded && visible.length < pairs.length
                ? `Show all ${pairs.length} params`
                : "Collapse"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Checkpoint browser
function CheckpointBrowser({
  runPath,
  onLoadInEvalRunner,
}: {
  runPath: string;
  onLoadInEvalRunner: (path: string) => void;
}) {
  const [checkpoints, setCheckpoints] = useState<DirEntry[] | null>(null);

  useEffect(() => {
    invoke<DirEntry[]>("list_dir", { path: `${runPath}/checkpoints` })
      .then((entries) =>
        setCheckpoints(
          entries
            .filter((e) => !e.is_dir && ["pt", "ckpt", "pth"].includes(e.extension))
            .sort((a, b) => a.name.localeCompare(b.name))
        )
      )
      .catch(() => setCheckpoints([]));
  }, [runPath]);

  if (!checkpoints?.length) return null;

  return (
    <div className="rounded-xl border border-canvas-border overflow-hidden">
      <div className="px-4 py-2.5 bg-canvas-surface">
        <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider">
          Checkpoints ({checkpoints.length})
        </p>
      </div>
      <div className="divide-y divide-canvas-border/40 bg-canvas-bg">
        {checkpoints.map((ckpt) => (
          <div key={ckpt.path} className="flex items-center gap-3 px-4 py-2">
            <span className="text-xs font-mono text-gray-300 flex-1 truncate">{ckpt.name}</span>
            <span className="text-xs text-canvas-muted shrink-0">
              {formatBytes(ckpt.size_bytes)}
            </span>
            <button
              onClick={() => onLoadInEvalRunner(ckpt.path)}
              className="btn-ghost text-xs shrink-0"
              title="Open this checkpoint in the Evaluation Runner"
            >
              Load in Eval Runner →
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Run panel (chart + hparams + checkpoints for one or more selected runs)
function RunPanel({
  run,
  metrics,
  color,
  onLoadCheckpoint,
}: {
  run: TrainingRun;
  metrics: TrainingMetricsRow[];
  color: string;
  onLoadCheckpoint: (path: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
        <p className="text-xs font-mono text-gray-300">{run.name}</p>
        <span className="text-xs text-canvas-muted">{metrics.length} epochs</span>
      </div>
      <GradNormSparkline metrics={metrics} />
      {run.has_hparams && <HparamsPanel runPath={run.path} />}
      <CheckpointBrowser runPath={run.path} onLoadInEvalRunner={onLoadCheckpoint} />
    </div>
  );
}

// ── Main page
export function TrainingMonitor() {
  const { projectRoot, setMode, setPendingCheckpoint } = useAppStore();
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metricsMap, setMetricsMap] = useState<Record<string, TrainingMetricsRow[]>>({});
  const [loading, setLoading] = useState(false);

  const logsPath = projectRoot ? `${projectRoot}/logs` : "";

  const discover = useCallback(async () => {
    if (!logsPath) return;
    setLoading(true);
    try {
      const found = await invoke<TrainingRun[]>("list_training_runs", { logsPath });
      setRuns(found);
    } finally {
      setLoading(false);
    }
  }, [logsPath]);

  useEffect(() => {
    if (logsPath) discover();
  }, [logsPath, discover]);

  const loadMetrics = useCallback(
    async (run: TrainingRun) => {
      if (metricsMap[run.name]) return;
      const rows = await invoke<TrainingMetricsRow[]>("load_training_metrics", {
        runPath: run.path,
      });
      setMetricsMap((m) => ({ ...m, [run.name]: rows }));
    },
    [metricsMap]
  );

  const toggleRun = useCallback(
    (run: TrainingRun) => {
      setSelected((s) =>
        s.includes(run.name) ? s.filter((r) => r !== run.name) : [...s, run.name]
      );
      loadMetrics(run);
    },
    [loadMetrics]
  );

  const selectedRunObjects = useMemo(
    () => runs.filter((r) => selected.includes(r.name)),
    [runs, selected]
  );

  const runsMetrics = useMemo(
    () =>
      selectedRunObjects
        .map((r) => ({ name: r.name, metrics: metricsMap[r.name] ?? [] }))
        .filter((rm) => rm.metrics.length > 0),
    [selectedRunObjects, metricsMap]
  );

  const handleLoadCheckpoint = useCallback(
    (checkpointPath: string) => {
      setPendingCheckpoint(checkpointPath);
      setMode("eval_runner");
    },
    [setPendingCheckpoint, setMode]
  );

  if (!projectRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <p className="text-sm">Project root not configured.</p>
        <p className="text-xs">Set the project root in Settings to discover training runs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={discover}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          Discover Runs
        </button>
        <span className="text-xs text-canvas-muted truncate">{logsPath}</span>
      </div>

      {runs.length === 0 && !loading && (
        <div className="card text-canvas-muted text-sm">
          No training runs found in{" "}
          <code className="font-mono text-xs">{logsPath}</code>.
        </div>
      )}

      {/* Run selector */}
      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-3">
            Training Runs — select to overlay
          </p>
          <div className="space-y-1">
            {runs.map((run, i) => {
              const color = RUN_COLORS[i % RUN_COLORS.length];
              return (
                <label
                  key={run.name}
                  className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-canvas-hover cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(run.name)}
                    onChange={() => toggleRun(run)}
                    className="accent-accent-primary"
                  />
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: selected.includes(run.name) ? color : "#444" }}
                  />
                  <span className="text-sm text-gray-300 font-mono flex-1 truncate">
                    {run.name}
                  </span>
                  <span className="flex gap-2 text-xs text-canvas-muted">
                    {run.has_metrics && <span className="text-accent-success">metrics</span>}
                    {run.has_hparams && <span>hparams</span>}
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      )}

      {/* Multi-run overlay chart */}
      {runsMetrics.length > 0 && <MultiRunChart runsMetrics={runsMetrics} />}

      {/* Per-run panels (grad norm, hparams, checkpoints) */}
      {selectedRunObjects.map((run, i) => {
        const metrics = metricsMap[run.name];
        if (!metrics) {
          return (
            <div key={run.name} className="card text-canvas-muted text-sm">
              Loading {run.name}…
            </div>
          );
        }
        return (
          <RunPanel
            key={run.name}
            run={run}
            metrics={metrics}
            color={RUN_COLORS[i % RUN_COLORS.length]}
            onLoadCheckpoint={handleLoadCheckpoint}
          />
        );
      })}
    </div>
  );
}
